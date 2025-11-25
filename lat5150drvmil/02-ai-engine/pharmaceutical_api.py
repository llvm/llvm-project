#!/usr/bin/env python3
"""
LAT5150DRVMIL Pharmaceutical Research API
FastAPI backend with TEMPEST security levels (0-3)

TEMPEST Compliance:
- Level 0 (PUBLIC): Basic properties, no auth
- Level 1 (RESTRICTED): ADMET, API key, 1000 req/day
- Level 2 (CONTROLLED): Docking, abuse potential, MFA, audit, 500 req/day
- Level 3 (CLASSIFIED): Patient simulation, gov auth, air-gap, ephemeral

Security: Localhost-only binding, no external access
"""

import os
import sys
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect, status
from fastapi.security import APIKeyHeader, HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add sub_agents to path
sys.path.insert(0, str(Path(__file__).parent / "sub_agents"))

# Import pharmaceutical corpus
from sub_agents.pharmaceutical_corpus import PharmaceuticalCorpus, TEMPESTLevel, AuditLogger


# =============================================================================
# TEMPEST Security Configuration
# =============================================================================

TEMPEST_CONFIG = {
    0: {  # PUBLIC
        "auth": None,
        "rate_limit": None,  # Unlimited
        "audit": False,
        "capabilities": ["structure", "descriptors", "basic_properties"]
    },
    1: {  # RESTRICTED
        "auth": "api_key",
        "rate_limit": 1000,  # requests per day
        "audit": False,
        "capabilities": ["admet_basic", "bbb_prediction", "drug_likeness", "classification"]
    },
    2: {  # CONTROLLED
        "auth": "mfa",  # Multi-factor (simulated via extended API key)
        "rate_limit": 500,
        "audit": True,
        "capabilities": ["docking", "abuse_potential", "admet_full", "safety_profile", "receptor_binding"]
    },
    3: {  # CLASSIFIED
        "auth": "government",  # Government authorization (simulated)
        "rate_limit": 100,
        "audit": True,
        "capabilities": ["patient_simulation", "protocol_optimization", "regulatory_dossier", "proactive_intel"]
    }
}


# =============================================================================
# Request/Response Models
# =============================================================================

class CompoundRequest(BaseModel):
    """Request to analyze a compound"""
    smiles: str = Field(..., description="SMILES string of the compound")
    name: Optional[str] = Field(None, description="Compound name (optional)")
    analysis_level: Optional[str] = Field("comprehensive", description="Analysis depth: basic, standard, comprehensive")

    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v) < 2:
            raise ValueError("SMILES string must be at least 2 characters")
        return v.strip()


class DockingRequest(BaseModel):
    """Request for molecular docking"""
    smiles: str = Field(..., description="SMILES string")
    receptors: Optional[List[str]] = Field(["MOR", "DOR", "KOR"], description="Target receptors")

    @validator('receptors')
    def validate_receptors(cls, v):
        valid = {"MOR", "DOR", "KOR", "NMDA", "5HT2A", "CB1", "CB2"}
        if not all(r in valid for r in v):
            raise ValueError(f"Invalid receptor. Valid options: {valid}")
        return v


class ADMETRequest(BaseModel):
    """Request for ADMET prediction"""
    smiles: str = Field(..., description="SMILES string")
    use_intel_ai: Optional[bool] = Field(True, description="Use Intel AI acceleration")
    cross_validate: Optional[bool] = Field(False, description="Cross-validate with RDKit")


class AbusePotentialRequest(BaseModel):
    """Request for abuse potential analysis"""
    smiles: str = Field(..., description="SMILES string")
    comprehensive: Optional[bool] = Field(False, description="12-hour comprehensive analysis (Level 3 only)")


class SimulationRequest(BaseModel):
    """Request for patient simulation"""
    protocol: Dict[str, Any] = Field(..., description="Treatment protocol configuration")
    n_patients: Optional[int] = Field(100000, description="Number of virtual patients")

    @validator('n_patients')
    def validate_patients(cls, v):
        if v < 100 or v > 1000000:
            raise ValueError("n_patients must be between 100 and 1,000,000")
        return v


class APIResponse(BaseModel):
    """Standard API response"""
    success: bool
    tempest_level: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: secrets.token_hex(8))


# =============================================================================
# Authentication & Authorization
# =============================================================================

class APIKeyManager:
    """Manage API keys with TEMPEST levels"""

    def __init__(self):
        self.keys = {
            # Level 0: No key needed
            # Level 1: RESTRICTED
            "pk_restricted_demo_1234567890abcdef": {
                "level": 1,
                "user_id": "demo_user",
                "rate_limit": 1000,
                "requests_today": 0,
                "last_reset": datetime.now()
            },
            # Level 2: CONTROLLED
            "pk_controlled_mfa_0987654321fedcba": {
                "level": 2,
                "user_id": "controlled_user",
                "rate_limit": 500,
                "requests_today": 0,
                "last_reset": datetime.now()
            },
            # Level 3: CLASSIFIED
            "pk_classified_gov_abcdef1234567890": {
                "level": 3,
                "user_id": "government_user",
                "rate_limit": 100,
                "requests_today": 0,
                "last_reset": datetime.now()
            }
        }

    def verify_key(self, api_key: str, required_level: int) -> Dict[str, Any]:
        """Verify API key and return user info"""
        # Level 0 requires no key
        if required_level == 0:
            return {"level": 0, "user_id": "public", "rate_limit": None}

        # Check if key exists
        if api_key not in self.keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        key_info = self.keys[api_key]

        # Check if key has sufficient level
        if key_info["level"] < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key level {key_info['level']} insufficient for TEMPEST Level {required_level}"
            )

        # Check rate limit
        now = datetime.now()
        if (now - key_info["last_reset"]).days >= 1:
            key_info["requests_today"] = 0
            key_info["last_reset"] = now

        if key_info["requests_today"] >= key_info["rate_limit"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded ({key_info['rate_limit']} requests/day)"
            )

        # Increment request count
        key_info["requests_today"] += 1

        return key_info


# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Global API key manager
api_key_manager = APIKeyManager()


async def verify_tempest_level(
    required_level: int,
    api_key: Optional[str] = Security(api_key_header)
) -> Dict[str, Any]:
    """Dependency to verify TEMPEST security level"""
    return api_key_manager.verify_key(api_key, required_level)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="LAT5150DRVMIL Pharmaceutical Research API",
    description="TEMPEST-compliant API for pharmaceutical research corpus",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware (localhost only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pharmaceutical corpus instances (one per TEMPEST level)
corpus_instances = {}


def get_corpus(tempest_level: int, user_id: str = "system") -> PharmaceuticalCorpus:
    """Get or create pharmaceutical corpus instance for TEMPEST level"""
    key = f"{tempest_level}_{user_id}"
    if key not in corpus_instances:
        corpus_instances[key] = PharmaceuticalCorpus(
            tempest_level=tempest_level,
            user_id=user_id
        )
    return corpus_instances[key]


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "LAT5150DRVMIL Pharmaceutical Research API",
        "version": "1.0.0",
        "tempest_compliant": True,
        "tempest_levels": {
            "0": "PUBLIC - Basic properties, no auth",
            "1": "RESTRICTED - ADMET, API key required",
            "2": "CONTROLLED - Docking, abuse potential, MFA required",
            "3": "CLASSIFIED - Patient simulation, gov auth required"
        },
        "endpoints": {
            "discovery": "/api/v1/discover",
            "validation": "/api/v2/validate",
            "safety": "/api/v2/safety",
            "optimization": "/api/v3/optimize",
            "reporting": "/api/v3/report"
        },
        "docs": "/api/docs",
        "status": "operational"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tempest_levels_available": [0, 1, 2, 3]
    }


@app.get("/api/tempest/info")
async def tempest_info():
    """Get TEMPEST level information"""
    return {
        "levels": TEMPEST_CONFIG,
        "description": {
            0: "PUBLIC - No authentication, basic molecular properties only",
            1: "RESTRICTED - API key required, ADMET predictions, 1000 req/day",
            2: "CONTROLLED - MFA required, docking + abuse potential, audit logging, 500 req/day",
            3: "CLASSIFIED - Government auth, patient simulation, air-gap support, 100 req/day"
        }
    }


# =============================================================================
# Level 0-1: Discovery Endpoints (PUBLIC/RESTRICTED)
# =============================================================================

@app.post("/api/v1/discover/screen", response_model=APIResponse)
async def screen_compound(
    request: CompoundRequest,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(1))
):
    """
    Screen compound for drug-likeness and therapeutic potential
    TEMPEST Level: 1 (RESTRICTED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.screen_compound(
            smiles=request.smiles,
            name=request.name,
            analysis_level=request.analysis_level
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


@app.post("/api/v1/discover/classify", response_model=APIResponse)
async def classify_compound(
    smiles: str,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(1))
):
    """
    Classify therapeutic potential (antidepressant, analgesic, etc.)
    TEMPEST Level: 1 (RESTRICTED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.classify_therapeutic_potential(smiles)

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


# =============================================================================
# Level 2: Validation Endpoints (CONTROLLED)
# =============================================================================

@app.post("/api/v2/validate/docking", response_model=APIResponse)
async def molecular_docking(
    request: DockingRequest,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(2))
):
    """
    Perform molecular docking to target receptors
    TEMPEST Level: 2 (CONTROLLED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.dock_to_receptors(
            smiles=request.smiles,
            receptors=request.receptors
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


@app.post("/api/v2/validate/admet", response_model=APIResponse)
async def admet_prediction(
    request: ADMETRequest,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(1))
):
    """
    Predict ADMET properties
    TEMPEST Level: 1 (RESTRICTED) - Basic
    TEMPEST Level: 2 (CONTROLLED) - Full toxicity
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.predict_admet(
            smiles=request.smiles,
            use_intel_ai=request.use_intel_ai,
            cross_validate=request.cross_validate
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


@app.post("/api/v2/validate/bbb", response_model=APIResponse)
async def bbb_prediction(
    smiles: str,
    cross_validate: bool = True,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(1))
):
    """
    Predict Blood-Brain Barrier penetration
    TEMPEST Level: 1 (RESTRICTED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.predict_bbb_penetration(
            smiles=smiles,
            cross_validate=cross_validate
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


# =============================================================================
# Level 2: Safety Assessment Endpoints (CONTROLLED)
# =============================================================================

@app.post("/api/v2/safety/profile", response_model=APIResponse)
async def safety_profile(
    smiles: str,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(2))
):
    """
    Generate comprehensive safety profile
    TEMPEST Level: 2 (CONTROLLED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.comprehensive_safety_profile(smiles)

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


@app.post("/api/v2/safety/abuse-potential", response_model=APIResponse)
async def abuse_potential(
    request: AbusePotentialRequest,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(2))
):
    """
    Predict abuse potential and NPS classification
    TEMPEST Level: 2 (CONTROLLED) - Standard analysis
    TEMPEST Level: 3 (CLASSIFIED) - Comprehensive 12-hour analysis
    """
    try:
        # Comprehensive analysis requires Level 3
        if request.comprehensive and auth["level"] < 3:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Comprehensive abuse analysis requires TEMPEST Level 3 (CLASSIFIED)"
            )

        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.predict_abuse_potential(
            smiles=request.smiles,
            comprehensive=request.comprehensive
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


# =============================================================================
# Level 3: Optimization Endpoints (CLASSIFIED)
# =============================================================================

@app.post("/api/v3/optimize/simulate", response_model=APIResponse)
async def patient_simulation(
    request: SimulationRequest,
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(3))
):
    """
    Run large-scale patient simulation
    TEMPEST Level: 3 (CLASSIFIED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.simulate_patients(
            compound_protocol=request.protocol,
            n_patients=request.n_patients
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


# =============================================================================
# Level 3: Reporting Endpoints (CLASSIFIED)
# =============================================================================

@app.post("/api/v3/report/dossier", response_model=APIResponse)
async def regulatory_dossier(
    smiles: str,
    format: str = "json",
    auth: Dict[str, Any] = Depends(lambda: verify_tempest_level(3))
):
    """
    Generate regulatory submission dossier
    TEMPEST Level: 3 (CLASSIFIED)
    """
    try:
        corpus = get_corpus(auth["level"], auth["user_id"])
        result = corpus.generate_regulatory_dossier(
            smiles=smiles,
            format=format
        )

        return APIResponse(
            success=True,
            tempest_level=auth["level"],
            data=result
        )

    except Exception as e:
        return APIResponse(
            success=False,
            tempest_level=auth["level"],
            error=str(e)
        )


# =============================================================================
# WebSocket for Real-Time Updates
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """
    WebSocket for real-time analysis updates
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now (can extend for real-time job monitoring)
            await websocket.send_json({
                "type": "echo",
                "message": data,
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LAT5150DRVMIL Pharmaceutical Research API")
    print("TEMPEST Compliance: Levels 0-3")
    print("=" * 70)
    print("\nSecurity: Localhost-only binding (127.0.0.1)")
    print("No external access permitted\n")
    print("TEMPEST Levels:")
    print("  0 (PUBLIC)     - No auth, basic properties")
    print("  1 (RESTRICTED) - API key, ADMET predictions")
    print("  2 (CONTROLLED) - MFA, docking, abuse potential, audit")
    print("  3 (CLASSIFIED) - Gov auth, patient simulation\n")
    print("Demo API Keys:")
    print("  Level 1: pk_restricted_demo_1234567890abcdef")
    print("  Level 2: pk_controlled_mfa_0987654321fedcba")
    print("  Level 3: pk_classified_gov_abcdef1234567890\n")
    print("API Docs: http://127.0.0.1:8000/api/docs")
    print("=" * 70)

    uvicorn.run(
        app,
        host="127.0.0.1",  # Localhost only - no external access
        port=8000,
        log_level="info"
    )
