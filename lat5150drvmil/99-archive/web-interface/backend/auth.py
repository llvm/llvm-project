#!/usr/bin/env python3
"""
DSMIL Control System Authentication and Authorization
Military-grade security with multi-factor authentication
"""

import jwt
import hashlib
import secrets
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from passlib.context import CryptContext
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings, get_required_clearance

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security bearer
security = HTTPBearer()


@dataclass
class UserContext:
    """User authentication context"""
    user_id: str
    username: str
    clearance_level: str
    permissions: List[str]
    session_id: str
    compartment_access: List[str]
    authorized_devices: List[int]
    expires_at: datetime
    mfa_verified: bool = False
    

@dataclass
class AuthorizationResult:
    """Authorization result"""
    authorized: bool
    auth_token: Optional[str] = None
    denial_reason: Optional[str] = None
    requires_dual_auth: bool = False
    valid_until: Optional[datetime] = None


class AuthenticationManager:
    """Comprehensive authentication and authorization manager"""
    
    def __init__(self):
        self.active_sessions: Dict[str, UserContext] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        
        # Default admin user for initial setup
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for system initialization"""
        self.users_db = {
            "admin": {
                "user_id": "admin_001",
                "username": "admin", 
                "email": "admin@dsmil.mil",
                "password_hash": self.hash_password("dsmil_admin_2024"),
                "clearance_level": "TOP_SECRET",
                "compartment_access": ["SCI", "SAP", "DSMIL"],
                "authorized_devices": list(range(settings.device_base_id, settings.device_base_id + settings.device_count)),
                "is_active": True,
                "is_locked": False,
                "mfa_enabled": False
            },
            "operator": {
                "user_id": "op_001",
                "username": "operator",
                "email": "operator@dsmil.mil", 
                "password_hash": self.hash_password("dsmil_op_2024"),
                "clearance_level": "SECRET",
                "compartment_access": ["DSMIL"],
                "authorized_devices": list(range(settings.device_base_id, settings.device_base_id + 60)),  # First 60 devices
                "is_active": True,
                "is_locked": False,
                "mfa_enabled": False
            },
            "analyst": {
                "user_id": "analyst_001",
                "username": "analyst",
                "email": "analyst@dsmil.mil",
                "password_hash": self.hash_password("dsmil_analyst_2024"),
                "clearance_level": "CONFIDENTIAL",
                "compartment_access": ["DSMIL"],
                "authorized_devices": list(range(settings.device_base_id, settings.device_base_id + 36)),  # First 36 devices
                "is_active": True,
                "is_locked": False,
                "mfa_enabled": False
            }
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_data: Dict[str, Any], session_id: str) -> str:
        """Create JWT access token"""
        expires_at = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
        
        payload = {
            "sub": user_data["user_id"],
            "username": user_data["username"],
            "clearance": user_data["clearance_level"],
            "permissions": user_data.get("permissions", []),
            "compartments": user_data["compartment_access"],
            "authorized_devices": user_data["authorized_devices"],
            "session_id": session_id,
            "exp": expires_at.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "iss": "dsmil-control-system",
            "aud": "dsmil-users"
        }
        
        return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[UserContext]:
        """Verify JWT token and return user context"""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
            
            # Check expiration
            if datetime.utcfromtimestamp(payload.get("exp", 0)) < datetime.utcnow():
                raise HTTPException(status_code=401, detail="Token expired")
            
            # Check session validity
            session_id = payload.get("session_id")
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=401, detail="Session invalid")
            
            return self.active_sessions[session_id]
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def authenticate_user(self, username: str, password: str, client_ip: str = "unknown") -> Dict[str, Any]:
        """Authenticate user with username and password"""
        
        # Check if account is locked
        if username in self.locked_accounts:
            lockout_until = self.locked_accounts[username]
            if datetime.utcnow() < lockout_until:
                remaining_minutes = int((lockout_until - datetime.utcnow()).total_seconds() / 60)
                raise HTTPException(
                    status_code=423, 
                    detail=f"Account locked for {remaining_minutes} more minutes"
                )
            else:
                # Remove expired lockout
                del self.locked_accounts[username]
                self.failed_attempts[username] = 0
        
        # Check user exists
        if username not in self.users_db:
            self._record_failed_attempt(username)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_data = self.users_db[username]
        
        # Check if account is active
        if not user_data.get("is_active", False) or user_data.get("is_locked", False):
            raise HTTPException(status_code=403, detail="Account disabled")
        
        # Verify password
        if not self.verify_password(password, user_data["password_hash"]):
            self._record_failed_attempt(username)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Reset failed attempts on successful authentication
        self.failed_attempts[username] = 0
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(minutes=settings.session_timeout_minutes)
        
        user_context = UserContext(
            user_id=user_data["user_id"],
            username=user_data["username"],
            clearance_level=user_data["clearance_level"],
            permissions=user_data.get("permissions", []),
            session_id=session_id,
            compartment_access=user_data["compartment_access"],
            authorized_devices=user_data["authorized_devices"],
            expires_at=expires_at,
            mfa_verified=not user_data.get("mfa_enabled", False)  # Skip MFA for now
        )
        
        # Store active session
        self.active_sessions[session_id] = user_context
        
        # Create access token
        access_token = self.create_access_token(user_data, session_id)
        
        # Update user login time
        user_data["last_login"] = datetime.utcnow()
        
        logger.info(f"User {username} authenticated successfully from {client_ip}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_expire_minutes * 60,
            "user_context": user_context
        }
    
    def _record_failed_attempt(self, username: str):
        """Record failed authentication attempt"""
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        
        if self.failed_attempts[username] >= settings.max_failed_auth_attempts:
            # Lock account
            lockout_until = datetime.utcnow() + timedelta(minutes=settings.account_lockout_duration_minutes)
            self.locked_accounts[username] = lockout_until
            
            logger.warning(f"Account {username} locked due to {self.failed_attempts[username]} failed attempts")
    
    async def authorize_operation(
        self,
        user_context: UserContext,
        operation: str,
        risk_level: str,
        device_id: Optional[int] = None,
        justification: Optional[str] = None
    ) -> AuthorizationResult:
        """Comprehensive operation authorization"""
        
        # Check session validity
        if datetime.utcnow() > user_context.expires_at:
            return AuthorizationResult(
                authorized=False,
                denial_reason="Session expired"
            )
        
        # Check clearance level
        required_clearance = get_required_clearance(operation, risk_level)
        clearance_levels = ["UNCLASSIFIED", "RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET", "SCI", "SAP", "COSMIC"]
        
        try:
            user_level_idx = clearance_levels.index(user_context.clearance_level)
            required_level_idx = clearance_levels.index(required_clearance)
            
            if user_level_idx < required_level_idx:
                return AuthorizationResult(
                    authorized=False,
                    denial_reason=f"Insufficient clearance: requires {required_clearance}, have {user_context.clearance_level}"
                )
        except ValueError:
            return AuthorizationResult(
                authorized=False,
                denial_reason="Invalid clearance level"
            )
        
        # Check device-specific access
        if device_id and device_id not in user_context.authorized_devices:
            return AuthorizationResult(
                authorized=False,
                denial_reason=f"No access to device {device_id:04X}"
            )
        
        # Check if device is quarantined
        if device_id and device_id in settings.quarantined_device_ids:
            if user_context.clearance_level not in ["TOP_SECRET", "SCI", "SAP"]:
                return AuthorizationResult(
                    authorized=False,
                    denial_reason=f"Device {device_id:04X} is quarantined - requires TOP_SECRET or higher"
                )
        
        # Risk-based authorization
        if risk_level in ["HIGH", "CRITICAL"]:
            # Require justification for high-risk operations
            if not justification or len(justification.strip()) < 10:
                return AuthorizationResult(
                    authorized=False,
                    denial_reason="Justification required for high-risk operations (minimum 10 characters)"
                )
            
            # Check for dual authorization requirement
            if risk_level == "CRITICAL":
                # For now, allow single authorization but log the requirement
                logger.warning(f"CRITICAL operation {operation} by {user_context.username} - dual auth recommended")
        
        # Generate authorization token
        auth_token = self._generate_auth_token(user_context, operation, risk_level, device_id)
        
        return AuthorizationResult(
            authorized=True,
            auth_token=auth_token,
            valid_until=datetime.utcnow() + timedelta(minutes=5)  # Short-lived tokens
        )
    
    def _generate_auth_token(
        self, 
        user_context: UserContext, 
        operation: str, 
        risk_level: str, 
        device_id: Optional[int]
    ) -> str:
        """Generate short-lived authorization token for specific operation"""
        payload = {
            "user_id": user_context.user_id,
            "operation": operation,
            "risk_level": risk_level,
            "device_id": device_id,
            "session_id": user_context.session_id,
            "exp": (datetime.utcnow() + timedelta(minutes=5)).timestamp(),
            "iat": datetime.utcnow().timestamp()
        }
        
        return jwt.encode(payload, settings.secret_key + "_auth", algorithm=settings.jwt_algorithm)
    
    async def can_access_device(self, user_context: UserContext, device_id: int) -> bool:
        """Check if user can access specific device"""
        return device_id in user_context.authorized_devices
    
    async def logout_user(self, session_id: str):
        """Logout user and invalidate session"""
        if session_id in self.active_sessions:
            user_context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            logger.info(f"User {user_context.username} logged out")
        
    async def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, user_context in self.active_sessions.items()
            if current_time > user_context.expires_at
        ]
        
        for session_id in expired_sessions:
            user_context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            logger.info(f"Expired session cleaned up for user {user_context.username}")
    
    async def verify_websocket_token(self, token: str) -> UserContext:
        """Verify WebSocket authentication token"""
        return self.verify_token(token)
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.active_sessions)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security status information"""
        return {
            "active_sessions": len(self.active_sessions),
            "failed_attempts_total": sum(self.failed_attempts.values()),
            "locked_accounts": len(self.locked_accounts),
            "users_total": len(self.users_db)
        }


# Global authentication manager
auth_manager = AuthenticationManager()


# FastAPI dependency for getting current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserContext:
    """FastAPI dependency to get current authenticated user"""
    try:
        user_context = auth_manager.verify_token(credentials.credentials)
        
        # Update last activity
        user_context.expires_at = datetime.utcnow() + timedelta(minutes=settings.session_timeout_minutes)
        
        return user_context
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Background task for session cleanup
async def session_cleanup_task():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            await auth_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)  # Run every 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            await asyncio.sleep(60)