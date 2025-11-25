#!/usr/bin/env python3
"""
DSMIL FastAPI Server - Type-Safe AI Inference with Automatic API Documentation

This is the next-generation API server using FastAPI with:
- Automatic OpenAPI/Swagger documentation
- Type-safe request/response with Pydantic
- Async/await support for better performance
- Built-in validation and error handling
- WebSocket support for streaming responses

Run: uvicorn dsmil_fastapi_server:app --host 127.0.0.1 --port 9877
Docs: http://localhost:9877/docs
"""

import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add AI engine to path
AI_ENGINE_DIR = Path(__file__).parent.parent / "02-ai-engine"
INTEGRATIONS_DIR = Path(__file__).parent.parent / "04-integrations"
sys.path.insert(0, str(AI_ENGINE_DIR))
sys.path.insert(0, str(INTEGRATIONS_DIR))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import DSMIL components
from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE
from unified_orchestrator import UnifiedAIOrchestrator

if PYDANTIC_AVAILABLE:
    from pydantic_models import (
        DSMILQueryRequest,
        DSMILQueryResult,
        OrchestratorRequest,
        OrchestratorResponse,
        RAGQueryRequest,
        RAGQueryResult,
        CodeGenerationResult,
        SecurityAnalysisResult,
        ModelTier,
        BackendType,
    )
    from rag_manager import RAGManager

    # WhiteRabbit integration
    try:
        from whiterabbit_pydantic import (
            PydanticWhiteRabbitEngine,
            WhiteRabbitRequest,
            WhiteRabbitResponse,
            WhiteRabbitDevice,
            WhiteRabbitTaskType,
            WHITERABBIT_AVAILABLE,
        )
    except ImportError:
        WHITERABBIT_AVAILABLE = False
        # Create stub classes to prevent NameError in endpoint definitions
        class WhiteRabbitRequest(BaseModel):
            """Stub for when WhiteRabbit not available"""
            prompt: str = Field(..., description="Input prompt")

        class WhiteRabbitResponse(BaseModel):
            """Stub for when WhiteRabbit not available"""
            response: str = Field(..., description="Generated response")
            error: str = Field(default="WhiteRabbit not available")
else:
    raise ImportError("Pydantic is required for FastAPI server. Install: pip install pydantic pydantic-ai")

# Create FastAPI app
app = FastAPI(
    title="DSMIL AI Engine API",
    description="""
    Hardware-Attested, Type-Safe AI Inference with Multi-Backend Orchestration

    ## Features

    - 🔒 **Hardware Attestation** - TPM-backed inference verification
    - 🎯 **Smart Routing** - Automatic model selection based on query complexity
    - 🔍 **Multi-Backend** - Local (Ollama), Gemini, OpenAI, specialized agents
    - 🌐 **Web Search** - Integrated DuckDuckGo and Shodan threat intelligence
    - 📚 **RAG System** - Document retrieval and knowledge base
    - ✅ **Type-Safe** - Full Pydantic validation with automatic API docs
    - 🚀 **High Performance** - Async/await, streaming responses

    ## Backends

    - **Local** - WhiteRabbitNeo-33B (primary), Qwen Coder (privacy-first, zero cost, NPU/GPU/NCS2)
    - **Gemini** - Google Gemini Pro (multimodal: images, video)
    - **OpenAI** - GPT-4, GPT-3.5 (explicit request only, structured outputs)
    - **Specialized** - Geospatial, RDKit, PRT, NMDA, NPS, Pharmaceutical

    ## Authentication

    Local-only by default (127.0.0.1). For remote access, use SSH tunneling.
    """,
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware (localhost only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
orchestrator: Optional[UnifiedAIOrchestrator] = None
ai_engine: Optional[DSMILAIEngine] = None
rag_manager: Optional[RAGManager] = None
whiterabbit_engine: Optional['PydanticWhiteRabbitEngine'] = None


@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup"""
    global orchestrator, ai_engine, rag_manager, whiterabbit_engine

    print("🚀 Initializing DSMIL AI Engine...")

    # Initialize with Pydantic mode enabled
    orchestrator = UnifiedAIOrchestrator(enable_ace=False, pydantic_mode=True)
    ai_engine = DSMILAIEngine(pydantic_mode=True)
    rag_manager = RAGManager(pydantic_mode=True)

    # Initialize WhiteRabbit if available
    if WHITERABBIT_AVAILABLE:
        try:
            whiterabbit_engine = PydanticWhiteRabbitEngine(pydantic_mode=True)
            print("✅ WhiteRabbit Engine initialized (NPU/GPU/NCS2 support)")
        except Exception as e:
            print(f"⚠  WhiteRabbit Engine initialization failed: {e}")
            whiterabbit_engine = None
    else:
        print("⚠  WhiteRabbit Engine not available")

    print("✅ DSMIL AI Engine ready")
    print(f"📖 API Documentation: http://127.0.0.1:9877/docs")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "DSMIL AI Engine",
        "version": "2.3.0",
        "status": "operational",
        "documentation": "/docs",
        "features": [
            "Hardware-attested AI inference",
            "Multi-backend orchestration",
            "Type-safe Pydantic models",
            "Smart routing with confidence scores",
            "Web search integration",
            "RAG document retrieval",
            "Streaming responses",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "orchestrator": orchestrator is not None,
        "ai_engine": ai_engine is not None,
        "rag_manager": rag_manager is not None,
        "whiterabbit_engine": whiterabbit_engine is not None,
    }


# ============================================================================
# AI Inference Endpoints
# ============================================================================

@app.post("/ai/chat", response_model=OrchestratorResponse, tags=["AI Inference"])
async def chat(request: OrchestratorRequest) -> OrchestratorResponse:
    """
    Main chat endpoint with smart routing and multi-backend support

    The orchestrator automatically routes queries to the optimal backend:
    - **Code queries** → Local coding models (DeepSeek Coder, Qwen Coder)
    - **General queries** → Fast local model (DeepSeek R1)
    - **Multimodal queries** → Gemini Pro (images, video)
    - **Current events** → Web search + local AI
    - **Threat intel** → Shodan + local AI
    - **Specialized** → Domain-specific agents (geospatial, chemistry, etc.)

    Returns fully validated OrchestratorResponse with routing metadata.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Convert force_backend enum to string if needed
        force_backend = request.force_backend.value if request.force_backend else None

        result = orchestrator.query(
            prompt=request.prompt,
            force_backend=force_backend,
            images=request.images,
            video=request.video,
        )

        # Result is already OrchestratorResponse (Pydantic mode enabled)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/ai/generate", response_model=DSMILQueryResult, tags=["AI Inference"])
async def generate(request: DSMILQueryRequest) -> DSMILQueryResult:
    """
    Direct AI generation endpoint (bypasses orchestrator)

    Use this for:
    - Direct model access without routing
    - Custom model selection
    - Fine-grained control over parameters

    Backends:
    - Local Ollama models (DeepSeek, Qwen, etc.)
    - Hardware attestation included
    """
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not initialized")

    try:
        result = ai_engine.generate(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/ai/simple", response_model=dict, tags=["AI Inference"])
async def simple_query(
    msg: str = Query(..., description="Message to send to AI", min_length=1),
    model: str = Query("auto", description="Model selection (auto, fast, code, quality_code)"),
) -> dict:
    """
    Simple query endpoint (backward compatible with http.server API)

    For quick queries without Pydantic models. Returns dict instead of validated model.
    Use /ai/chat for type-safe responses.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        force_backend = None if model == "auto" else model

        # Get dict response (orchestrator in Pydantic mode returns OrchestratorResponse)
        result = orchestrator.query(msg, force_backend=force_backend)

        # Convert Pydantic model to dict for backward compatibility
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================================================
# RAG Endpoints
# ============================================================================

@app.post("/rag/search", response_model=RAGQueryResult, tags=["RAG System"])
async def rag_search(request: RAGQueryRequest) -> RAGQueryResult:
    """
    Search RAG knowledge base

    Performs semantic search over ingested documents and returns
    ranked results with relevance scores and metadata.
    """
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG manager not initialized")

    try:
        results = rag_manager.search(request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")


@app.post("/rag/add-file", tags=["RAG System"])
async def rag_add_file(
    filepath: str = Query(..., description="Path to file to ingest"),
    background_tasks: BackgroundTasks = None,
):
    """
    Add file to RAG knowledge base

    Supported formats: PDF, TXT, MD, LOG, C, H, PY, SH, CPP, JAVA
    File is chunked, embedded, and indexed for semantic search.
    """
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG manager not initialized")

    try:
        result = rag_manager.add_file(filepath)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


@app.get("/rag/stats", tags=["RAG System"])
async def rag_stats():
    """Get RAG system statistics"""
    if not rag_manager:
        raise HTTPException(status_code=503, detail="RAG manager not initialized")

    try:
        return rag_manager.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


# ============================================================================
# WhiteRabbit Endpoints
# ============================================================================

@app.post("/whiterabbit/generate", response_model=WhiteRabbitResponse, tags=["WhiteRabbit"])
async def whiterabbit_generate(request: WhiteRabbitRequest) -> WhiteRabbitResponse:
    """
    Direct WhiteRabbitNeo inference with multi-device support

    WhiteRabbitNeo is the primary local AI model with:
    - **Multi-Device Support** - NPU, GPU (Arc), NCS2, or CPU
    - **Dynamic Quantization** - Auto INT4/INT8/FP16 selection
    - **Dual-Model Validation** - Secondary model validates outputs
    - **Hardware Attestation** - TPM-backed inference verification

    Devices:
    - **NPU** - Intel Arc NPU (fastest, lowest power)
    - **GPU_ARC** - Intel Arc iGPU (balanced performance)
    - **NCS2** - Intel Neural Compute Stick 2 (portable)
    - **SYSTEM** - CPU fallback (compatibility)
    - **AUTO** - Automatic device selection (recommended)

    Task Types:
    - **Text Generation** - General queries, conversations
    - **Code Generation** - Code synthesis and completion
    - **Code Review** - Security analysis, bug detection
    - **Summarization** - Document and text summarization
    - **Q&A** - Question answering
    """
    if not WHITERABBIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="WhiteRabbit not available. Install dependencies: pip install pydantic pydantic-ai && ollama pull whiterabbit-neo-33b"
        )

    if not whiterabbit_engine:
        raise HTTPException(
            status_code=503,
            detail="WhiteRabbit engine not initialized. Check logs for errors."
        )

    try:
        result = whiterabbit_engine.generate(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"WhiteRabbit inference failed: {str(e)}"
        )


@app.get("/whiterabbit/status", tags=["WhiteRabbit"])
async def whiterabbit_status():
    """Get WhiteRabbit engine status and available devices"""
    if not WHITERABBIT_AVAILABLE:
        return {
            "available": False,
            "reason": "WhiteRabbit dependencies not installed",
            "install_command": "pip install pydantic pydantic-ai && ollama pull whiterabbit-neo-33b"
        }

    if not whiterabbit_engine:
        return {
            "available": False,
            "reason": "WhiteRabbit engine not initialized"
        }

    try:
        # Get available devices from WhiteRabbit engine
        return {
            "available": True,
            "engine": "WhiteRabbitNeo-33B",
            "devices": ["npu", "gpu_arc", "ncs2", "system"],
            "default_device": "auto",
            "capabilities": [
                "Multi-device inference",
                "Dynamic quantization",
                "Dual-model validation",
                "Hardware attestation",
                "Device-aware smart routing",
                "NCS2 memory pooling",
            ]
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


# ============================================================================
# Status & Configuration Endpoints
# ============================================================================

@app.get("/status", tags=["System"])
async def system_status():
    """
    Get comprehensive system status

    Returns information about all available backends, models,
    and specialized agents.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        status = orchestrator.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.get("/models", tags=["System"])
async def list_models():
    """List all available AI models across all backends"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        status = orchestrator.get_status()
        return {
            "local": status["backends"]["local_deepseek"]["models"],
            "gemini": status["backends"]["gemini_pro"]["model"],
            "openai": status["backends"]["openai_pro"]["models"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9877, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     DSMIL AI Engine - FastAPI Server                                 ║
║     Type-Safe API with Automatic Documentation                       ║
╚══════════════════════════════════════════════════════════════════════╝

Starting server on http://{args.host}:{args.port}

📖 API Documentation:
   - Swagger UI:  http://{args.host}:{args.port}/docs
   - ReDoc:       http://{args.host}:{args.port}/redoc
   - OpenAPI:     http://{args.host}:{args.port}/openapi.json

🔒 Security: Localhost-only by default (use SSH tunnel for remote access)
    """)

    uvicorn.run(
        "dsmil_fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
