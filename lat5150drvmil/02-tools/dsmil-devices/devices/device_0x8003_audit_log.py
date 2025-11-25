#!/usr/bin/env python3
"""
Device 0x8003: Audit Log

Security audit logging and compliance tracking for all DSMIL operations.
Maintains tamper-proof audit trail for forensic analysis and compliance.

Device ID: 0x8003
Group: 0 (Core Security)
Risk Level: SAFE (Read-only audit access)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class AuditSeverity:
    """Audit log severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    SECURITY = 3
    CRITICAL = 4


class AuditCategory:
    """Audit log categories"""
    SYSTEM = 0
    SECURITY = 1
    ACCESS = 2
    CONFIGURATION = 3
    NETWORK = 4
    STORAGE = 5
    AUTHENTICATION = 6


class AuditLogDevice(DSMILDeviceBase):
    """Audit Log (0x8003)"""

    # Register map
    REG_LOG_STATUS = 0x00
    REG_LOG_COUNT = 0x04
    REG_LOG_CAPACITY = 0x08
    REG_LOG_POSITION = 0x0C
    REG_LAST_SEVERITY = 0x10
    REG_TAMPER_STATUS = 0x14

    # Status bits
    STATUS_LOG_ACTIVE = 0x01
    STATUS_LOG_FULL = 0x02
    STATUS_TAMPER_DETECTED = 0x04
    STATUS_SEALED = 0x08
    STATUS_ENCRYPTED = 0x10
    STATUS_REMOTE_SYNC = 0x20

    def __init__(self, device_id: int = 0x8003,
                 name: str = "Audit Log",
                 description: str = "Security Audit Logging and Compliance Tracking"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.audit_entries = []
        self.max_entries = 10000
        self.sealed = False
        self.encrypted = True
        self.tamper_detected = False

        # Initialize with some sample entries
        self._initialize_sample_entries()

        # Register map
        self.register_map = {
            "LOG_STATUS": {
                "offset": self.REG_LOG_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Audit log status"
            },
            "LOG_COUNT": {
                "offset": self.REG_LOG_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of audit entries"
            },
            "LOG_CAPACITY": {
                "offset": self.REG_LOG_CAPACITY,
                "size": 4,
                "access": "RO",
                "description": "Maximum audit log capacity"
            },
            "LOG_POSITION": {
                "offset": self.REG_LOG_POSITION,
                "size": 4,
                "access": "RO",
                "description": "Current log write position"
            },
            "LAST_SEVERITY": {
                "offset": self.REG_LAST_SEVERITY,
                "size": 4,
                "access": "RO",
                "description": "Last entry severity level"
            },
            "TAMPER_STATUS": {
                "offset": self.REG_TAMPER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Tamper detection status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Audit Log device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize audit log
            self._initialize_sample_entries()
            self.sealed = False
            self.encrypted = True
            self.tamper_detected = False

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "log_entries": len(self.audit_entries),
                "capacity": self.max_entries,
                "sealed": self.sealed,
                "encrypted": self.encrypted,
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.EVENT_MONITORING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "log_active": bool(status_reg & self.STATUS_LOG_ACTIVE),
            "log_full": bool(status_reg & self.STATUS_LOG_FULL),
            "tamper_detected": bool(status_reg & self.STATUS_TAMPER_DETECTED),
            "sealed": bool(status_reg & self.STATUS_SEALED),
            "encrypted": bool(status_reg & self.STATUS_ENCRYPTED),
            "remote_sync": bool(status_reg & self.STATUS_REMOTE_SYNC),
            "entry_count": len(self.audit_entries),
            "capacity": self.max_entries,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "LOG_STATUS":
                value = self._read_status_register()
            elif register == "LOG_COUNT":
                value = len(self.audit_entries)
            elif register == "LOG_CAPACITY":
                value = self.max_entries
            elif register == "LOG_POSITION":
                value = len(self.audit_entries)
            elif register == "LAST_SEVERITY":
                value = self.audit_entries[-1]['severity'] if self.audit_entries else 0
            elif register == "TAMPER_STATUS":
                value = 1 if self.tamper_detected else 0
            else:
                value = 0

            self._record_operation(True)
            return OperationResult(True, data={
                "register": register,
                "value": value,
                "hex": f"0x{value:08X}",
            })

        except Exception as e:
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    # Audit Log specific operations

    def get_recent_entries(self, limit: int = 50) -> OperationResult:
        """Get recent audit entries"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        entries = self.audit_entries[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "entries": entries,
            "total": len(entries),
            "showing": f"Last {min(limit, len(self.audit_entries))} of {len(self.audit_entries)}",
        })

    def get_entries_by_severity(self, severity: int) -> OperationResult:
        """Get audit entries by severity level"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        entries = [e for e in self.audit_entries if e['severity'] == severity]

        self._record_operation(True)
        return OperationResult(True, data={
            "entries": entries,
            "total": len(entries),
            "severity": self._get_severity_name(severity),
        })

    def get_entries_by_category(self, category: int) -> OperationResult:
        """Get audit entries by category"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        entries = [e for e in self.audit_entries if e['category'] == category]

        self._record_operation(True)
        return OperationResult(True, data={
            "entries": entries,
            "total": len(entries),
            "category": self._get_category_name(category),
        })

    def get_summary(self) -> OperationResult:
        """Get audit log summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Count by severity
        by_severity = {}
        for entry in self.audit_entries:
            sev = entry['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1

        # Count by category
        by_category = {}
        for entry in self.audit_entries:
            cat = entry['category']
            by_category[cat] = by_category.get(cat, 0) + 1

        summary = {
            "total_entries": len(self.audit_entries),
            "capacity": self.max_entries,
            "utilization": f"{len(self.audit_entries)/self.max_entries*100:.1f}%",
            "by_severity": {
                self._get_severity_name(sev): count
                for sev, count in by_severity.items()
            },
            "by_category": {
                self._get_category_name(cat): count
                for cat, count in by_category.items()
            },
            "sealed": self.sealed,
            "encrypted": self.encrypted,
            "tamper_detected": self.tamper_detected,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        stats = super().get_statistics()

        stats.update({
            "total_entries": len(self.audit_entries),
            "capacity": self.max_entries,
            "utilization_percent": len(self.audit_entries)/self.max_entries*100,
            "sealed": self.sealed,
            "encrypted": self.encrypted,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read audit log status register (simulated)"""
        status = self.STATUS_LOG_ACTIVE

        if len(self.audit_entries) >= self.max_entries:
            status |= self.STATUS_LOG_FULL

        if self.tamper_detected:
            status |= self.STATUS_TAMPER_DETECTED

        if self.sealed:
            status |= self.STATUS_SEALED

        if self.encrypted:
            status |= self.STATUS_ENCRYPTED

        return status

    def _initialize_sample_entries(self):
        """Initialize with sample audit entries"""
        current_time = time.time()

        self.audit_entries = [
            {
                "timestamp": current_time - 3600,
                "severity": AuditSeverity.INFO,
                "category": AuditCategory.SYSTEM,
                "message": "System boot completed",
                "user": "system",
                "device": "0x8001",
            },
            {
                "timestamp": current_time - 3000,
                "severity": AuditSeverity.SECURITY,
                "category": AuditCategory.AUTHENTICATION,
                "message": "User authentication successful",
                "user": "admin",
                "device": "0x8002",
            },
            {
                "timestamp": current_time - 2400,
                "severity": AuditSeverity.INFO,
                "category": AuditCategory.ACCESS,
                "message": "Device 0x8005 initialized",
                "user": "system",
                "device": "0x8005",
            },
            {
                "timestamp": current_time - 1800,
                "severity": AuditSeverity.WARNING,
                "category": AuditCategory.SECURITY,
                "message": "Multiple failed authentication attempts",
                "user": "unknown",
                "device": "0x8002",
            },
            {
                "timestamp": current_time - 600,
                "severity": AuditSeverity.INFO,
                "category": AuditCategory.CONFIGURATION,
                "message": "Configuration updated",
                "user": "admin",
                "device": "0x8000",
            },
        ]

    def _get_severity_name(self, severity: int) -> str:
        """Get severity level name"""
        names = {
            AuditSeverity.DEBUG: "Debug",
            AuditSeverity.INFO: "Info",
            AuditSeverity.WARNING: "Warning",
            AuditSeverity.SECURITY: "Security",
            AuditSeverity.CRITICAL: "Critical",
        }
        return names.get(severity, "Unknown")

    def _get_category_name(self, category: int) -> str:
        """Get category name"""
        names = {
            AuditCategory.SYSTEM: "System",
            AuditCategory.SECURITY: "Security",
            AuditCategory.ACCESS: "Access",
            AuditCategory.CONFIGURATION: "Configuration",
            AuditCategory.NETWORK: "Network",
            AuditCategory.STORAGE: "Storage",
            AuditCategory.AUTHENTICATION: "Authentication",
        }
        return names.get(category, "Unknown")
