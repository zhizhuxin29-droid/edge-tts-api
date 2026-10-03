#!/usr/bin/env python3
"""
Edge TTS API 服务
基于 edge-tts 库的 FastAPI 服务
启动: python3 edge_tts_api.py
"""

import asyncio
import io
import os
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import edge_tts

# ============ 配置 ============
CACHE_DIR = "/tmp/edge_tts_cache"
MAX_CACHE_SIZE_MB = 500  # 缓存上限 MB
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

os.makedirs(CACHE_DIR, exist_ok=True)

# 可用语音列表（常用）
VOICE_LIST = {
    # 中文
    "zh-CN-XiaoxiaoNeural": "晓晓（女，温柔）",
    "zh-CN-XiaoyiNeural": "晓伊（女，活泼）",
    "zh-CN-YunjianNeural": "云健（男，新闻）",
    "zh-CN-YunxiNeural": "云希（男，少年）",
    "zh-CN-YunxiaNeural": "云夏（男，童声）",
    "zh-CN-YunyangNeural": "云扬（男，解说）",
    "zh-CN-liaoning-XiaobeiNeural": "晓北（东北话）",
    "zh-CN-shaanxi-XiaoniNeural": "晓妮（陕西话）",
    "zh-TW-HsiaoChenNeural": "晓臻（台湾，女）",
    "zh-TW-YunJheNeural": "云哲（台湾，男）",
    # 英文
    "en-US-AriaNeural": "Aria（美，女）",
    "en-US-GuyNeural": "Guy（美，男）",
    "en-GB-SoniaNeural": "Sonia（英，女）",
    "en-GB-RyanNeural": "Ryan（英，男）",
    # 日文
    "ja-JP-NanamiNeural": "七海（女）",
    "ja-JP-KeitaNeural": "圭太（男）",
    # 韩文
    "ko-KR-SunHiNeural": "善熙（女）",
    # 粤语
    "zh-HK-HiuMaanNeural": "晓曼（粤语，女）",
    "zh-HK-WanLungNeural": "云龙（粤语，男）",
    # 泰文
    "th-TH-PremwadeeNeural": "Premwadee(泰语，女)",
    "th-TH-NiwatNeural": "Niwat(泰语，男)"
}

# ============ 模型定义 ============
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="要合成的文本")
    voice: str = Field(default=DEFAULT_VOICE, description="语音名称")
    rate: str = Field(default="+0%", description="语速调整，如 +20% 或 -10%")
    volume: str = Field(default="+0%", description="音量调整，如 +20% 或 -10%")
    pitch: str = Field(default="+0Hz", description="音调调整，如 +50Hz")
    format: str = Field(default="mp3", description="输出格式: mp3, wav, ogg, webm")

class TTSResponse(BaseModel):
    success: bool
    message: str
    voice: str
    text_length: int
    duration_ms: Optional[int] = None
    cache_hit: bool = False

# ============ 缓存管理 ============
def get_cache_key(text: str, voice: str, rate: str, volume: str, pitch: str, fmt: str) -> str:
    """生成缓存文件名"""
    content = f"{text}|{voice}|{rate}|{volume}|{pitch}|{fmt}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cache_path(cache_key: str, fmt: str) -> str:
    return os.path.join(CACHE_DIR, f"{cache_key}.{fmt}")

def clean_cache():
    """清理过期缓存"""
    try:
        files = []
        for f in os.listdir(CACHE_DIR):
            fp = os.path.join(CACHE_DIR, f)
            if os.path.isfile(fp):
                files.append((fp, os.path.getsize(fp), os.path.getmtime(fp)))

        total_size = sum(f[1] for f in files) / (1024 * 1024)
        if total_size > MAX_CACHE_SIZE_MB:
            # 按时间排序，删除最旧的
            files.sort(key=lambda x: x[2])
            for fp, size, _ in files:
                if total_size <= MAX_CACHE_SIZE_MB * 0.7:
                    break
                os.remove(fp)
                total_size -= size / (1024 * 1024)
    except Exception:
        pass

# ============ FastAPI 应用 ============
app = FastAPI(
    title="Edge TTS API",
    description="基于 edge-tts 的文本转语音 API 服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "Edge TTS API",
        "version": "1.0.0",
        "endpoints": {
            "GET /voices": "获取可用语音列表",
            "POST /tts": "文本转语音（返回音频流）",
            "GET /tts": "文本转语音（GET方式，返回音频流）",
            "GET /health": "健康检查",
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "cache_dir": CACHE_DIR}

@app.get("/voices")
async def list_voices():
    """获取可用语音列表"""
    return {
        "count": len(VOICE_LIST),
        "voices": [{"id": k, "name": v} for k, v in VOICE_LIST.items()]
    }

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """
    文本转语音（POST方式）
    直接返回音频流
    """
    if req.voice not in VOICE_LIST:
        raise HTTPException(status_code=400, detail=f"不支持的语音: {req.voice}")

    cache_key = get_cache_key(req.text, req.voice, req.rate, req.volume, req.pitch, req.format)
    cache_path = get_cache_path(cache_key, req.format)

    # 检查缓存
    if os.path.exists(cache_path):
        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
        }.get(req.format, "audio/mpeg")

        return FileResponse(
            cache_path,
            media_type=media_type,
            headers={"X-Cache": "HIT", "X-Cache-Key": cache_key}
        )

    # 生成音频
    try:
        communicate = edge_tts.Communicate(
            req.text,
            req.voice,
            rate=req.rate,
            volume=req.volume,
            pitch=req.pitch,
        )

        # 收集音频数据
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_data = audio_buffer.getvalue()

        # 写入缓存
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
            clean_cache()
        except Exception:
            pass

        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
        }.get(req.format, "audio/mpeg")

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={"X-Cache": "MISS", "X-Cache-Key": cache_key}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 生成失败: {str(e)}")

@app.get("/tts")
async def text_to_speech_get(
    text: str = Query(..., min_length=1, max_length=5000),
    voice: str = Query(default=DEFAULT_VOICE),
    rate: str = Query(default="+0%"),
    volume: str = Query(default="+0%"),
    pitch: str = Query(default="+0Hz"),
    format: str = Query(default="mp3"),
):
    """
    文本转语音（GET方式，方便浏览器直接访问）
    示例: /tts?text=你好世界&voice=zh-CN-XiaoxiaoNeural
    """
    req = TTSRequest(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        format=format,
    )
    return await text_to_speech(req)

# ============ 启动 ============
if __name__ == "__main__:":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
