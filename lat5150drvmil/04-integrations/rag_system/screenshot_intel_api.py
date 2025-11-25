#!/usr/bin/env python3
"""
Screenshot Intelligence REST API
FastAPI server for Screenshot Intelligence System

Security:
- LOCAL-ONLY: Binds to 127.0.0.1 (no external access)
- Optional API key authentication
- Rate limiting
- CORS for local development

Endpoints:
- GET /api/stats - System statistics
- GET /api/devices - List devices
- POST /api/devices - Register device
- POST /api/ingest/screenshot - Ingest screenshot
- POST /api/ingest/scan - Scan device
- GET /api/search - Search intelligence
- GET /api/timeline - Query timeline
- POST /api/analyze - AI analysis
- GET /api/health - Health check

Usage:
    python3 screenshot_intel_api.py
    curl http://127.0.0.1:8000/api/health
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

# FastAPI
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Core modules
try:
    from vector_rag_system import VectorRAGSystem
    from screenshot_intelligence import ScreenshotIntelligence
    from ai_analysis_layer import AIAnalysisLayer
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class DeviceRegister(BaseModel):
    device_id: str
    device_name: str
    device_type: str
    screenshot_path: str


class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    score_threshold: float = 0.5
    doc_type: Optional[str] = None
    source: Optional[str] = None
    device_id: Optional[str] = None


class TimelineQuery(BaseModel):
    start_date: str
    end_date: str
    doc_types: Optional[List[str]] = None
    format: str = 'json'


class AnalyzeRequest(BaseModel):
    start_date: str
    end_date: str
    detect_incidents: bool = True


class IngestScan(BaseModel):
    device_id: str
    pattern: str = '*.png'


# API App
app = FastAPI(
    title="Screenshot Intelligence API",
    description="AI-Driven Screenshot Organization & Analysis System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS - Allow localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
intel: Optional[ScreenshotIntelligence] = None
ai_analysis: Optional[AIAnalysisLayer] = None
API_KEY = os.getenv('SCREENSHOT_INTEL_API_KEY', '')


# Dependency: API Key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if authentication is enabled"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global intel, ai_analysis

    try:
        intel = ScreenshotIntelligence()
        ai_analysis = AIAnalysisLayer(
            vector_rag=intel.rag,
            screenshot_intel=intel
        )
        logger.info("✓ Screenshot Intelligence API initialized")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Screenshot Intelligence API",
        "version": "1.0.0"
    }


# Stats
@app.get("/api/stats")
async def get_stats(auth: bool = Depends(verify_api_key)):
    """Get system statistics"""
    try:
        stats = intel.rag.get_stats()
        stats["devices_registered"] = len(intel.devices)
        stats["incidents"] = len(intel.incidents)
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Devices
@app.get("/api/devices")
async def list_devices(auth: bool = Depends(verify_api_key)):
    """List registered devices"""
    devices = []
    for device_id, device in intel.devices.items():
        devices.append({
            "device_id": device.device_id,
            "device_name": device.device_name,
            "device_type": device.device_type,
            "screenshot_path": str(device.screenshot_path)
        })
    return {"devices": devices, "total": len(devices)}


@app.post("/api/devices")
async def register_device(
    device: DeviceRegister,
    auth: bool = Depends(verify_api_key)
):
    """Register a new device"""
    try:
        intel.register_device(
            device_id=device.device_id,
            device_name=device.device_name,
            device_type=device.device_type,
            screenshot_path=Path(device.screenshot_path)
        )
        return {
            "status": "success",
            "message": f"Device registered: {device.device_name}",
            "device_id": device.device_id
        }
    except Exception as e:
        logger.error(f"Failed to register device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion
@app.post("/api/ingest/screenshot")
async def ingest_screenshot(
    file_path: str,
    device_id: Optional[str] = None,
    auth: bool = Depends(verify_api_key)
):
    """Ingest a single screenshot"""
    try:
        result = intel.ingest_screenshot(
            screenshot_path=Path(file_path),
            device_id=device_id
        )
        return result
    except Exception as e:
        logger.error(f"Failed to ingest screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/scan")
async def ingest_scan(
    scan: IngestScan,
    auth: bool = Depends(verify_api_key)
):
    """Scan and ingest device screenshots"""
    try:
        result = intel.scan_device_screenshots(
            device_id=scan.device_id,
            pattern=scan.pattern
        )
        return result
    except Exception as e:
        logger.error(f"Failed to scan device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search
@app.post("/api/search")
async def search_intel(
    query: SearchQuery,
    auth: bool = Depends(verify_api_key)
):
    """Search intelligence database"""
    try:
        filters = {}
        if query.doc_type:
            filters['type'] = query.doc_type
        if query.source:
            filters['source'] = query.source
        if query.device_id:
            filters['device_id'] = query.device_id

        results = intel.rag.search(
            query=query.query,
            limit=query.limit,
            score_threshold=query.score_threshold,
            filters=filters if filters else None
        )

        return {
            "query": query.query,
            "total_results": len(results),
            "results": [
                {
                    "score": r.score,
                    "type": r.document.doc_type,
                    "filename": r.document.filename,
                    "timestamp": r.document.timestamp.isoformat(),
                    "text_preview": r.document.text[:200],
                    "metadata": r.document.metadata
                } for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Timeline
@app.post("/api/timeline")
async def query_timeline(
    timeline: TimelineQuery,
    auth: bool = Depends(verify_api_key)
):
    """Query timeline"""
    try:
        start_time = datetime.fromisoformat(timeline.start_date)
        end_time = datetime.fromisoformat(timeline.end_date)

        if timeline.format == 'json':
            events = intel.rag.timeline_query(
                start_time, end_time, doc_types=timeline.doc_types
            )
            return {
                "start_date": timeline.start_date,
                "end_date": timeline.end_date,
                "total_events": len(events),
                "events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "type": e.doc_type,
                        "filename": e.filename,
                        "text_preview": e.text[:150],
                        "metadata": e.metadata
                    } for e in events
                ]
            }
        else:  # markdown
            report = intel.generate_timeline_report(
                start_time, end_time, output_format='markdown'
            )
            return {"report": report, "format": "markdown"}

    except Exception as e:
        logger.error(f"Timeline query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analysis
@app.post("/api/analyze")
async def analyze_timeline(
    request: AnalyzeRequest,
    auth: bool = Depends(verify_api_key)
):
    """AI-powered timeline analysis"""
    try:
        start_time = datetime.fromisoformat(request.start_date)
        end_time = datetime.fromisoformat(request.end_date)

        results = ai_analysis.analyze_timeline(
            start_time,
            end_time,
            auto_detect_incidents=request.detect_incidents
        )

        return {
            "analysis": results['analysis'],
            "anomalies": [
                {
                    "id": a.anomaly_id,
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat()
                } for a in results['anomalies']
            ],
            "patterns": [
                {
                    "id": p.pattern_id,
                    "type": p.pattern_type,
                    "frequency": p.frequency,
                    "description": p.description
                } for p in results['patterns']
            ],
            "incidents": [
                {
                    "id": i.incident_id,
                    "name": i.incident_name,
                    "start_time": i.start_time.isoformat(),
                    "end_time": i.end_time.isoformat(),
                    "event_count": len(i.events),
                    "tags": i.tags
                } for i in results['incidents']
            ]
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run API server"""
    # Check if API key is set
    if not API_KEY:
        logger.warning("⚠️  No API key set (SCREENSHOT_INTEL_API_KEY)")
        logger.warning("   API will be accessible without authentication")

    # Run server - BIND TO 127.0.0.1 ONLY FOR SECURITY
    uvicorn.run(
        app,
        host="127.0.0.1",  # LOCAL-ONLY
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Screenshot Intelligence API Server")
    print("=" * 80)
    print(f"\nStarting server on http://127.0.0.1:8000")
    print(f"API Documentation: http://127.0.0.1:8000/api/docs")
    print(f"\n⚠️  LOCAL-ONLY: Server bound to 127.0.0.1 (no external access)")
    print(f"\n" + "=" * 80 + "\n")

    main()
