#!/usr/bin/env python3
"""
DSMIL Control System Configuration
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "DSMIL Control System"
    app_version: str = "1.0.0"
    debug_mode: bool = False
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_hosts: List[str] = ["localhost", "127.0.0.1", "dsmil-control.mil.local"]
    
    # Security
    secret_key: str = "dsmil-control-system-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    cors_origins: List[str] = [
        "http://localhost:3000", 
        "https://localhost:3000",
        "https://dsmil-control.mil.local"
    ]
    
    # Database
    database_url: str = "postgresql+asyncpg://dsmil:dsmil_secure_2024@localhost:5432/dsmil_control"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Existing SQLite database (for compatibility)
    sqlite_database_path: str = "/home/john/LAT5150DRVMIL/database/data/dsmil_tokens.db"
    
    # DSMIL Kernel Module
    kernel_module_path: str = "/dev/dsmil_control"
    sysfs_device_path: str = "/sys/class/dsmil"
    
    # Device configuration
    device_base_id: int = 0x8000  # 32768
    device_count: int = 84
    device_groups: int = 7  # 6 regular groups + 1 quarantined group
    devices_per_group: int = 12
    
    # Critical quarantined devices - NEVER ACCESS (NSA confirmed destruction capability)
    quarantined_device_ids: List[int] = [
        0x8009,  # Emergency Wipe Controller - EXTREME DANGER
        0x800A,  # Secondary Wipe Trigger - EXTREME DANGER  
        0x800B,  # Final Sanitization - EXTREME DANGER
        0x8019,  # Network Isolation/Wipe - HIGH DANGER
        0x8029   # Communications Blackout - HIGH DANGER
    ]
    
    # Phase 1 safe monitoring devices (NSA 65-90% confidence)
    safe_monitoring_ids: List[int] = [
        # Original proven safe (100% confidence)
        0x8000, 0x8001, 0x8002, 0x8003, 0x8004, 0x8006,
        # NSA-identified safe (65-90% confidence) 
        0x8007, 0x8010, 0x8012, 0x8015, 0x8016,
        0x8020, 0x8021, 0x8023, 0x8024, 0x8025,
        # JRTC1 Training controllers (50-60% confidence)
        0x8060, 0x8061, 0x8062, 0x8063, 0x8064, 0x8065,
        0x8066, 0x8067, 0x8068, 0x8069, 0x806A, 0x806B
    ]
    
    # Security thresholds
    max_failed_auth_attempts: int = 3
    account_lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    
    # Audit settings
    audit_retention_days: int = 365
    max_audit_entries: int = 1000000
    
    # Performance monitoring
    metrics_collection_interval: int = 60  # seconds
    performance_alert_threshold: int = 200  # milliseconds
    
    # Emergency settings
    emergency_stop_timeout: int = 5  # seconds
    emergency_cooldown_minutes: int = 15
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # WebSocket configuration
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    max_websocket_connections: int = 100
    
    # Thermal monitoring
    thermal_warning_threshold: int = 85  # Celsius
    thermal_critical_threshold: int = 95
    thermal_emergency_threshold: int = 100
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_operations_per_minute: int = 20
    
    class Config:
        env_file = ".env"
        env_prefix = "DSMIL_"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Device configuration helpers
def get_device_info(device_id: int) -> dict:
    """Get device information from device ID"""
    if not (settings.device_base_id <= device_id < settings.device_base_id + settings.device_count):
        raise ValueError(f"Invalid device ID: {device_id}")
    
    device_index = device_id - settings.device_base_id
    group_number = device_index // settings.devices_per_group
    device_in_group = device_index % settings.devices_per_group
    
    # Determine risk level based on group and quarantine status
    if device_id in settings.quarantined_device_ids:
        risk_level = "QUARANTINED"
        status = "QUARANTINED"
    elif group_number == 0:  # Master control group
        risk_level = "CRITICAL"
        status = "ACTIVE"
    elif group_number <= 2:  # Core system groups
        risk_level = "HIGH" 
        status = "ACTIVE"
    elif group_number <= 4:  # Standard operation groups
        risk_level = "MODERATE"
        status = "ACTIVE"
    else:  # Auxiliary groups
        risk_level = "LOW"
        status = "ACTIVE"
    
    return {
        "device_id": device_id,
        "device_name": f"DSMIL-{group_number:02d}-{device_in_group:02d}",
        "device_group": group_number,
        "device_index": device_in_group,
        "risk_level": risk_level,
        "status": status,
        "is_quarantined": device_id in settings.quarantined_device_ids,
        "hex_id": f"0x{device_id:04X}",
        "description": get_device_description(group_number, device_in_group)
    }


def get_device_description(group: int, device: int) -> str:
    """Get human-readable device description based on NSA intelligence"""
    # Specific device descriptions from NSA assessment
    specific_devices = {
        0x8000: "Group 0 Controller - Master coordination",
        0x8001: "Thermal Monitoring - Temperature sensors",
        0x8002: "Power Status - Power management monitoring",
        0x8003: "Fan Control - Cooling system management",
        0x8004: "CPU Status - Processor state monitoring",
        0x8006: "System Supervisor - Overall system health",
        0x8007: "Security Audit Logger - DoD audit trails",
        0x8009: "QUARANTINED - Emergency Wipe Controller",
        0x800A: "QUARANTINED - Secondary Wipe Trigger",
        0x800B: "QUARANTINED - Final Sanitization",
        0x8010: "Multi-Factor Authentication - CAC/PIV cards",
        0x8012: "Security Event Correlator - Real-time analysis",
        0x8015: "Certificate Authority Interface - PKI validation",
        0x8016: "Security Baseline Monitor - Drift detection",
        0x8019: "QUARANTINED - Network Isolation/Wipe",
        0x8020: "Network Interface Controller - Ethernet/WiFi",
        0x8021: "Wireless Communication Manager - WiFi/BT/Cell",
        0x8023: "Network Performance Monitor - Real-time metrics",
        0x8024: "VPN Hardware Accelerator - IPSec/SSL VPN",
        0x8025: "Network Quality of Service - Traffic shaping",
        0x8029: "QUARANTINED - Communications Blackout"
    }
    
    device_id = 0x8000 + (group * 12) + device
    if device_id in specific_devices:
        return specific_devices[device_id]
    
    # Training devices
    if 0x8060 <= device_id <= 0x8063:
        return f"Training Scenario Controller {device_id - 0x8060}"
    elif 0x8064 <= device_id <= 0x8067:
        return f"Training Data Collection {device_id - 0x8064}"
    elif 0x8068 <= device_id <= 0x806B:
        return f"Training Environment Control {device_id - 0x8068}"
    
    # Generic group descriptions
    group_descriptions = {
        0: "Core Security & Emergency",
        1: "Extended Security",
        2: "Network & Communications", 
        3: "Data Processing (UNKNOWN)",
        4: "Storage Control (UNKNOWN)",
        5: "Peripheral Management (UNKNOWN)",
        6: "Training Functions (JRTC1)"
    }
    
    group_name = group_descriptions.get(group, f"Unknown Group {group}")
    return f"{group_name} - Device {device:02d}"


def is_device_quarantined(device_id: int) -> bool:
    """Check if device is quarantined - CRITICAL SAFETY CHECK"""
    return device_id in settings.quarantined_device_ids

def is_device_safe_monitored(device_id: int) -> bool:
    """Check if device is in Phase 1 safe monitoring list"""
    return device_id in settings.safe_monitoring_ids

def get_device_access_level(device_id: int) -> str:
    """Get device access level based on NSA assessment"""
    if is_device_quarantined(device_id):
        return "NEVER_ACCESS"
    elif is_device_safe_monitored(device_id):
        return "READ_ONLY"
    else:
        return "RESTRICTED"


def get_all_device_configs() -> List[dict]:
    """Get configuration for all devices"""
    devices = []
    for device_id in range(settings.device_base_id, settings.device_base_id + settings.device_count):
        devices.append(get_device_info(device_id))
    return devices


# Risk assessment helpers
def assess_operation_risk(device_id: int, operation_type: str, operation_data: dict = None) -> str:
    """Assess risk level of an operation"""
    device_info = get_device_info(device_id)
    base_risk = device_info["risk_level"]
    
    # Risk escalation based on operation type
    risk_escalation = {
        "READ": 0,
        "WRITE": 1,
        "CONFIG": 2,
        "RESET": 3,
        "ACTIVATE": 2,
        "DEACTIVATE": 1
    }
    
    risk_levels = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL", "QUARANTINED"]
    
    try:
        current_level = risk_levels.index(base_risk)
        escalation = risk_escalation.get(operation_type.upper(), 0)
        final_level = min(current_level + escalation, len(risk_levels) - 1)
        return risk_levels[final_level]
    except (ValueError, KeyError):
        return "HIGH"  # Default to high risk for unknown operations


# Security configuration
def get_required_clearance(operation_type: str, risk_level: str) -> str:
    """Get required clearance level for operation"""
    clearance_matrix = {
        ("READ", "LOW"): "RESTRICTED",
        ("READ", "MODERATE"): "CONFIDENTIAL", 
        ("READ", "HIGH"): "SECRET",
        ("READ", "CRITICAL"): "SECRET",
        ("WRITE", "LOW"): "CONFIDENTIAL",
        ("WRITE", "MODERATE"): "SECRET",
        ("WRITE", "HIGH"): "SECRET",
        ("WRITE", "CRITICAL"): "TOP_SECRET",
        ("CONFIG", "MODERATE"): "SECRET",
        ("CONFIG", "HIGH"): "TOP_SECRET",
        ("CONFIG", "CRITICAL"): "TOP_SECRET",
        ("RESET", "HIGH"): "TOP_SECRET",
        ("RESET", "CRITICAL"): "TOP_SECRET",
        ("ACTIVATE", "HIGH"): "SECRET",
        ("ACTIVATE", "CRITICAL"): "TOP_SECRET"
    }
    
    return clearance_matrix.get((operation_type.upper(), risk_level.upper()), "TOP_SECRET")