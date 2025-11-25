#!/usr/bin/env python3
"""
DSMIL Control System FastAPI Backend Server
Military-grade device control and monitoring API

This server provides secure access to 84 DSMIL devices through a RESTful API
with real-time WebSocket updates and comprehensive audit logging.
"""

import asyncio
import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Local imports
from config import settings
from models import database, init_db
from auth import auth_manager, get_current_user
from device_controller import device_controller
from websocket_manager import websocket_manager
from audit_logger import audit_logger
from security_monitor import security_monitor
from emergency_stop import emergency_stop

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("DSMIL Control System starting up...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize device controller
        await device_controller.initialize()
        logger.info("Device controller initialized")
        
        # Initialize security monitor
        await security_monitor.initialize()
        logger.info("Security monitor initialized")
        
        # Start background tasks
        asyncio.create_task(security_monitor.monitor_loop())
        asyncio.create_task(device_controller.health_check_loop())
        
        logger.info("DSMIL Control System startup complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
    
    finally:
        logger.info("DSMIL Control System shutting down...")
        
        # Cleanup
        await device_controller.cleanup()
        await security_monitor.cleanup()
        await websocket_manager.cleanup()
        
        logger.info("DSMIL Control System shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="DSMIL Control System API",
    description="Military-grade DSMIL device control and monitoring API",
    version="1.0.0",
    docs_url="/api/v1/docs" if settings.debug_mode else None,
    redoc_url="/api/v1/redoc" if settings.debug_mode else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)


# Request/Response Models
class DeviceOperationRequest(BaseModel):
    device_id: int = Field(..., ge=0x8000, le=0x806B, description="DSMIL device ID (32768-32875)")
    operation_type: str = Field(..., description="Operation type (READ, WRITE, CONFIG, RESET)")
    operation_data: Optional[Dict[str, Any]] = Field(None, description="Operation-specific data")
    justification: Optional[str] = Field(None, min_length=10, max_length=512, description="Operation justification")
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        allowed_ops = ['READ', 'WRITE', 'CONFIG', 'RESET', 'ACTIVATE', 'DEACTIVATE']
        if v.upper() not in allowed_ops:
            raise ValueError(f'Operation type must be one of: {allowed_ops}')
        return v.upper()
    
    class Config:
        schema_extra = {
            "example": {
                "device_id": 32768,  # 0x8000
                "operation_type": "READ",
                "operation_data": {"register": "STATUS", "offset": 0},
                "justification": "Routine health check of master security controller"
            }
        }


class DeviceOperationResponse(BaseModel):
    operation_id: str
    device_id: int
    operation_type: str
    result: str  # SUCCESS, DENIED, ERROR, EMERGENCY_STOP
    data: Optional[Dict[str, Any]]
    timestamp: str
    user_id: str
    audit_trail_id: str
    execution_time_ms: Optional[int] = None


class SystemStatusResponse(BaseModel):
    timestamp: str
    overall_status: str  # NORMAL, WARNING, CRITICAL, EMERGENCY
    device_count: int
    active_devices: int
    quarantined_devices: List[int]
    system_health: Dict[str, Any]
    security_status: Dict[str, Any]
    performance_metrics: Dict[str, float]


class EmergencyStopRequest(BaseModel):
    justification: str = Field(..., min_length=10, max_length=512, description="Emergency stop justification")


# Middleware for request tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID and response time tracking"""
    import time
    import uuid
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(round(process_time * 1000, 2))
    
    return response


# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "version": "1.0.0"
    }


# System status endpoints
@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    current_user = Depends(get_current_user)
) -> SystemStatusResponse:
    """Get comprehensive system status"""
    
    try:
        # Check authorization
        if not await auth_manager.authorize_operation(
            current_user, "SYSTEM_STATUS_READ", "LOW"
        ):
            raise HTTPException(status_code=403, detail="Insufficient privileges")
        
        # Gather system metrics
        device_states = await device_controller.get_all_device_states()
        system_health = await security_monitor.get_system_health()
        security_status = await security_monitor.get_security_status()
        performance_metrics = await device_controller.get_performance_metrics()
        
        # Log access
        await audit_logger.log_event(
            event_type="SYSTEM_STATUS_ACCESS",
            user_id=current_user.user_id,
            risk_level="LOW",
            details={"clearance_level": current_user.clearance_level}
        )
        
        return SystemStatusResponse(
            timestamp=str(asyncio.get_event_loop().time()),
            overall_status=system_health.get("overall_status", "UNKNOWN"),
            device_count=len(device_states),
            active_devices=sum(1 for state in device_states if state.get("active", False)),
            quarantined_devices=[d["device_id"] for d in device_states if d.get("quarantined", False)],
            system_health=system_health,
            security_status=security_status,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


# Device management endpoints
@app.get("/api/v1/devices")
async def list_devices(
    current_user = Depends(get_current_user),
    include_quarantined: bool = False
):
    """List all accessible devices based on user clearance"""
    
    try:
        # Get all devices
        all_devices = await device_controller.get_device_registry()
        
        # Filter based on user clearance and permissions
        accessible_devices = []
        for device in all_devices:
            if await auth_manager.can_access_device(current_user, device["device_id"]):
                if not device.get("quarantined", False) or include_quarantined:
                    accessible_devices.append(device)
        
        # Log device listing access
        await audit_logger.log_event(
            event_type="DEVICE_LIST_ACCESS",
            user_id=current_user.user_id,
            details={
                "total_devices": len(accessible_devices),
                "include_quarantined": include_quarantined
            }
        )
        
        return accessible_devices
        
    except Exception as e:
        logger.error(f"Failed to list devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve device list")


@app.post("/api/v1/devices/{device_id}/operations", response_model=DeviceOperationResponse)
async def execute_device_operation(
    device_id: int,
    request: DeviceOperationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
) -> DeviceOperationResponse:
    """Execute operation on specified device"""
    
    # Validate device ID matches request
    if device_id != request.device_id:
        raise HTTPException(status_code=400, detail="Device ID mismatch")
    
    # Check emergency stop
    if await emergency_stop.is_active():
        raise HTTPException(status_code=503, detail="System in emergency stop mode")
    
    try:
        # Risk assessment
        operation_risk = await device_controller.assess_operation_risk(
            device_id, request.operation_type, request.operation_data
        )
        
        # Authorization check
        auth_result = await auth_manager.authorize_operation(
            current_user, 
            f"DEVICE_{request.operation_type}",
            operation_risk,
            device_id=device_id,
            justification=request.justification
        )
        
        if not auth_result["authorized"]:
            await audit_logger.log_event(
                event_type="OPERATION_DENIED",
                user_id=current_user.user_id,
                device_id=device_id,
                risk_level=operation_risk,
                details={"denial_reason": auth_result["denial_reason"]}
            )
            raise HTTPException(status_code=403, detail=auth_result["denial_reason"])
        
        # Execute operation
        operation_result = await device_controller.execute_operation(
            device_id=device_id,
            operation_type=request.operation_type,
            operation_data=request.operation_data,
            auth_token=auth_result["auth_token"],
            user_context=current_user
        )
        
        # Log successful operation
        audit_entry = await audit_logger.log_event(
            event_type="DEVICE_OPERATION",
            user_id=current_user.user_id,
            device_id=device_id,
            operation_type=request.operation_type,
            risk_level=operation_risk,
            result="SUCCESS",
            details={
                "operation_data": request.operation_data,
                "justification": request.justification,
                "execution_time_ms": operation_result["execution_time_ms"]
            }
        )
        
        # Schedule background notification
        background_tasks.add_task(
            websocket_manager.broadcast_device_update,
            device_id,
            operation_result
        )
        
        return DeviceOperationResponse(
            operation_id=operation_result["operation_id"],
            device_id=device_id,
            operation_type=request.operation_type,
            result="SUCCESS",
            data=operation_result.get("result_data"),
            timestamp=str(asyncio.get_event_loop().time()),
            user_id=current_user.user_id,
            audit_trail_id=audit_entry["entry_id"],
            execution_time_ms=operation_result.get("execution_time_ms")
        )
        
    except Exception as e:
        # Log failed operation
        await audit_logger.log_event(
            event_type="DEVICE_OPERATION_FAILED",
            user_id=current_user.user_id,
            device_id=device_id,
            operation_type=request.operation_type,
            result="ERROR",
            details={"error": str(e)}
        )
        logger.error(f"Operation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")


# Emergency stop endpoint
@app.post("/api/v1/emergency-stop")
async def trigger_emergency_stop(
    request: EmergencyStopRequest,
    current_user = Depends(get_current_user)
):
    """Trigger system-wide emergency stop"""
    
    try:
        # Emergency stop should be available to all authenticated users
        await emergency_stop.activate(
            reason=request.justification,
            triggered_by=current_user.user_id
        )
        
        await audit_logger.log_event(
            event_type="EMERGENCY_STOP_TRIGGERED",
            user_id=current_user.user_id,
            risk_level="CRITICAL",
            details={"justification": request.justification}
        )
        
        # Broadcast emergency stop to all connected clients
        await websocket_manager.broadcast_emergency_stop(request.justification)
        
        logger.warning(f"Emergency stop activated by user {current_user.user_id}: {request.justification}")
        
        return {
            "message": "Emergency stop activated", 
            "timestamp": str(asyncio.get_event_loop().time())
        }
        
    except Exception as e:
        logger.error(f"Failed to activate emergency stop: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate emergency stop")


# WebSocket endpoint for real-time updates
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket endpoint for real-time system updates"""
    
    await websocket.accept()
    connection_id = None
    
    try:
        # Authenticate WebSocket connection
        if token:
            user_context = await auth_manager.verify_websocket_token(token)
            connection_id = await websocket_manager.add_connection(websocket, user_context)
            logger.info(f"WebSocket connection established for user {user_context.user_id}")
        else:
            await websocket.close(code=1008, reason="Authentication required")
            return
        
        # Keep connection alive and handle messages
        async for message in websocket.iter_text():
            try:
                import json
                data = json.loads(message)
                await websocket_manager.handle_message(connection_id, data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "ERROR",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "ERROR",
                    "message": "Message processing failed"
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if connection_id:
            await websocket_manager.remove_connection(connection_id)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower(),
        access_log=True
    )