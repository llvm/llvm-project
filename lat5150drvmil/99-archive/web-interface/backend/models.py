#!/usr/bin/env python3
"""
DSMIL Control System Database Models
SQLAlchemy models for PostgreSQL backend
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Boolean, Text, BigInteger, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func

Base = declarative_base()


class DeviceRegistry(Base):
    """Device registry with complete device information"""
    __tablename__ = "device_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, unique=True, index=True, nullable=False)
    device_name = Column(String(255), nullable=False)
    device_group = Column(Integer, nullable=False)
    device_index = Column(Integer, nullable=False)
    
    # Risk and security information
    risk_level = Column(String(20), nullable=False, index=True)
    security_classification = Column(String(50))
    required_clearance = Column(String(20))
    compartment_access = Column(ARRAY(String), default=[])
    
    # Device capabilities and constraints
    capabilities = Column(JSON)
    constraints = Column(JSON)
    hardware_info = Column(JSON)
    
    # Operational status
    is_active = Column(Boolean, default=False, index=True)
    is_quarantined = Column(Boolean, default=False, index=True)
    last_accessed = Column(DateTime)
    access_count = Column(BigInteger, default=0)
    error_count = Column(BigInteger, default=0)
    success_count = Column(BigInteger, default=0)
    
    # Performance metrics
    average_response_time_ms = Column(Float)
    last_response_time_ms = Column(Float)
    uptime_percentage = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    operations = relationship("OperationLog", back_populates="device")
    audit_entries = relationship("AuditLog", back_populates="device")


class UserAccount(Base):
    """User account information"""
    __tablename__ = "user_accounts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)
    username = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(255))
    
    # Security information
    password_hash = Column(String(255), nullable=False)
    clearance_level = Column(String(20), nullable=False)
    compartment_access = Column(ARRAY(String), default=[])
    authorized_devices = Column(ARRAY(Integer), default=[])
    
    # Account status
    is_active = Column(Boolean, default=True, index=True)
    is_locked = Column(Boolean, default=False, index=True)
    failed_login_attempts = Column(Integer, default=0)
    last_login = Column(DateTime)
    last_failed_login = Column(DateTime)
    lockout_until = Column(DateTime)
    
    # MFA settings
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(32))
    backup_codes = Column(ARRAY(String))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user")
    audit_entries = relationship("AuditLog", back_populates="user")


class UserSession(Base):
    """User session management"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    
    # Session details
    login_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_activity = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    client_ip = Column(String(45))  # IPv6 support
    user_agent = Column(Text)
    
    # Session state
    is_active = Column(Boolean, default=True, index=True)
    mfa_verified = Column(Boolean, default=False)
    operations_count = Column(Integer, default=0)
    last_operation_at = Column(DateTime)
    
    # Security flags
    suspicious_activity = Column(Boolean, default=False)
    force_logout = Column(Boolean, default=False)
    logout_reason = Column(String(255))
    logout_time = Column(DateTime)
    
    # Relationships
    user = relationship("UserAccount", back_populates="sessions")


class AuditLog(Base):
    """Comprehensive audit log for all system activities"""
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    sequence_number = Column(BigInteger, nullable=False, index=True, default=func.nextval('audit_sequence'))
    
    # Temporal information
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    session_id = Column(String(128), index=True)
    
    # User context
    user_id = Column(String(64), nullable=False, index=True)
    username = Column(String(128))
    user_clearance = Column(String(20))
    user_ip_address = Column(String(45))
    
    # Operation details
    event_type = Column(String(50), nullable=False, index=True)
    device_id = Column(Integer, index=True)
    operation_type = Column(String(50))
    risk_level = Column(String(20), index=True)
    
    # Authorization and result
    authorization_result = Column(String(20), nullable=False, index=True)
    operation_result = Column(String(20), index=True)
    denial_reason = Column(Text)
    
    # Detailed information
    operation_details = Column(JSON)
    system_context = Column(JSON)
    error_details = Column(JSON)
    
    # Security and compliance
    integrity_hash = Column(String(64), nullable=False)
    chain_hash = Column(String(64))
    compliance_flags = Column(Integer, default=0)
    external_reference = Column(String(128))
    
    # Performance metrics
    execution_time_ms = Column(Integer)
    system_load = Column(Float)
    memory_usage_mb = Column(Integer)
    
    # Relationships
    device = relationship("DeviceRegistry", back_populates="audit_entries")
    user = relationship("UserAccount", back_populates="audit_entries")


class OperationLog(Base):
    """Device operation execution log"""
    __tablename__ = "operation_log"
    
    id = Column(Integer, primary_key=True, index=True)
    operation_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    
    # Basic operation info
    device_id = Column(Integer, nullable=False, index=True)
    operation_type = Column(String(50), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    session_id = Column(String(128))
    
    # Timing information
    requested_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    execution_time_ms = Column(Integer)
    queue_time_ms = Column(Integer)
    
    # Operation data and results
    operation_data = Column(JSON)
    result_data = Column(JSON)
    status = Column(String(20), nullable=False, index=True)  # PENDING, EXECUTING, SUCCESS, FAILED
    error_message = Column(Text)
    
    # Authorization information
    auth_token_id = Column(String(128))
    dual_auth_required = Column(Boolean, default=False)
    dual_auth_completed = Column(Boolean, default=False)
    dual_auth_user_id = Column(String(64))
    justification = Column(Text)
    
    # Risk and safety
    assessed_risk = Column(String(20), index=True)
    safety_checks_passed = Column(Boolean, default=True)
    risk_mitigation_applied = Column(JSON)
    
    # Performance metrics
    cpu_usage_percent = Column(Integer)
    memory_usage_kb = Column(Integer)
    io_operations = Column(Integer)
    network_latency_ms = Column(Integer)
    
    # Relationships
    device = relationship("DeviceRegistry", back_populates="operations")


class SecurityEvent(Base):
    """Security events and threat detection"""
    __tablename__ = "security_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    
    # Event classification
    event_type = Column(String(50), nullable=False, index=True)
    severity_level = Column(String(20), nullable=False, index=True)
    threat_level = Column(String(20), index=True)
    confidence_score = Column(Integer)  # 0-100
    
    # Temporal information
    detected_at = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    event_count = Column(Integer, default=1)
    
    # Source information
    source_user_id = Column(String(64), index=True)
    source_ip = Column(String(45), index=True)
    source_device_id = Column(Integer, index=True)
    source_session_id = Column(String(128))
    
    # Event details
    event_description = Column(Text, nullable=False)
    event_data = Column(JSON)
    indicators_of_compromise = Column(ARRAY(String))
    attack_vector = Column(String(100))
    
    # Impact assessment
    affected_systems = Column(ARRAY(String))
    potential_damage_level = Column(String(20))
    business_impact = Column(String(20))
    
    # Response information
    response_status = Column(String(20), default="DETECTED", index=True)
    automated_response = Column(Boolean, default=False)
    response_actions = Column(JSON)
    assigned_analyst = Column(String(64))
    escalated_to = Column(String(64))
    
    # Resolution
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    false_positive = Column(Boolean, default=False, index=True)
    lessons_learned = Column(Text)


class PerformanceMetrics(Base):
    """System performance metrics collection"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Temporal information
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    collection_interval_seconds = Column(Integer, default=60)
    
    # System metrics
    cpu_usage_percent = Column(Float)
    memory_usage_percent = Column(Float)
    memory_total_mb = Column(Integer)
    memory_available_mb = Column(Integer)
    disk_usage_percent = Column(Float)
    disk_total_gb = Column(Float)
    disk_available_gb = Column(Float)
    
    # Thermal metrics
    system_temperature_celsius = Column(Float)
    cpu_temperature_celsius = Column(Float)
    thermal_throttling = Column(Boolean, default=False)
    
    # System load
    system_load_1min = Column(Float)
    system_load_5min = Column(Float)
    system_load_15min = Column(Float)
    
    # Network metrics
    network_connections_active = Column(Integer)
    network_packets_sent = Column(BigInteger)
    network_packets_received = Column(BigInteger)
    network_errors = Column(Integer)
    
    # DSMIL-specific metrics
    active_devices = Column(Integer)
    total_operations = Column(BigInteger)
    operations_per_second = Column(Float)
    average_operation_latency_ms = Column(Float)
    error_rate_percent = Column(Float)
    successful_operations = Column(BigInteger)
    failed_operations = Column(BigInteger)
    
    # Security metrics
    authentication_attempts = Column(Integer)
    failed_authentications = Column(Integer)
    security_events_count = Column(Integer)
    high_severity_events = Column(Integer)
    active_threats = Column(Integer)
    
    # Database performance
    db_connections_active = Column(Integer)
    db_connections_idle = Column(Integer)
    db_query_average_time_ms = Column(Float)
    db_transaction_rate = Column(Float)
    db_deadlocks = Column(Integer)
    
    # Application metrics
    websocket_connections = Column(Integer)
    api_requests_per_minute = Column(Integer)
    cache_hit_rate_percent = Column(Float)


class SystemConfiguration(Base):
    """System configuration parameters"""
    __tablename__ = "system_configuration"
    
    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(JSON)
    config_type = Column(String(20))  # STRING, INTEGER, BOOLEAN, JSON, ARRAY
    description = Column(Text)
    
    # Security and access
    is_sensitive = Column(Boolean, default=False)
    required_clearance = Column(String(20))
    modification_log = Column(JSON)
    
    # Versioning
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(64))


class EmergencyStop(Base):
    """Emergency stop events and status"""
    __tablename__ = "emergency_stops"
    
    id = Column(Integer, primary_key=True, index=True)
    stop_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Event details
    triggered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    triggered_by = Column(String(64), nullable=False)
    trigger_reason = Column(Text, nullable=False)
    trigger_type = Column(String(20), nullable=False)  # USER, AUTOMATIC, SYSTEM
    
    # Status tracking
    is_active = Column(Boolean, default=True, index=True)
    resolved_at = Column(DateTime)
    resolved_by = Column(String(64))
    resolution_notes = Column(Text)
    
    # Impact assessment
    affected_devices = Column(ARRAY(Integer))
    active_operations_stopped = Column(Integer)
    system_state_before = Column(JSON)
    system_state_after = Column(JSON)
    
    # Recovery information
    recovery_started_at = Column(DateTime)
    recovery_completed_at = Column(DateTime)
    recovery_time_minutes = Column(Integer)
    recovery_notes = Column(Text)


# Database helper functions
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    async def init_database(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)


# Global database instance (to be initialized)
database: Optional[DatabaseManager] = None


async def init_db():
    """Initialize database connection"""
    global database
    from config import settings
    
    database = DatabaseManager(settings.database_url)
    await database.init_database()
    return database