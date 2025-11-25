#!/usr/bin/env python3
"""
TPM2 Security Audit Logger
Comprehensive security event logging with military-grade audit trails

Author: TPM2 Security Audit Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import signal
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging.handlers

class SecurityEventType(Enum):
    """Security event types"""
    TOKEN_VALIDATION = "token_validation"
    AUTHORIZATION_GRANTED = "authorization_granted"
    AUTHORIZATION_DENIED = "authorization_denied"
    AUTHORIZATION_ERROR = "authorization_error"
    ME_SESSION_ESTABLISHED = "me_session_established"
    ME_SESSION_CLOSED = "me_session_closed"
    TPM_COMMAND_PROCESSED = "tpm_command_processed"
    ACCELERATION_USAGE = "acceleration_usage"
    FALLBACK_ACTIVATION = "fallback_activation"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_CONDITION = "error_condition"

class SecurityLevel(Enum):
    """Security levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SecurityEvent:
    """Security audit event"""
    event_id: str
    timestamp: float
    event_type: SecurityEventType
    security_level: SecurityLevel
    source: str
    user_id: Optional[str]
    process_id: int
    session_id: Optional[str]
    description: str
    details: Dict[str, Any]
    outcome: str  # SUCCESS, FAILURE, ERROR
    classification: str
    checksum: str

@dataclass
class AuditSession:
    """Audit session tracking"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    user_id: Optional[str]
    process_id: int
    events_count: int
    last_activity: float

class SecurityAuditLogger:
    """
    Enterprise-grade security audit logger for TPM2 operations
    Provides tamper-evident audit trails with military-grade security
    """

    def __init__(self, config_path: str = "/etc/military-tmp/audit.json"):
        """Initialize security audit logger"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.active_sessions = {}
        self.event_queue = []
        self.queue_lock = threading.Lock()
        self.logger_thread = None
        self.sequence_number = 0
        self.session_counter = 0

        # Setup structured logging
        self._setup_logging()

        # Initialize audit log
        self._initialize_audit_log()

        logger = logging.getLogger(__name__)
        logger.info("Security Audit Logger initialized")

    def start(self):
        """Start audit logging"""
        logger = logging.getLogger(__name__)
        logger.info("Starting security audit logging...")

        self.running = True

        # Log system startup
        self.log_event(
            SecurityEventType.SYSTEM_STARTUP,
            SecurityLevel.MEDIUM,
            "Security audit logging started",
            {"config_path": self.config_path}
        )

        # Start logging thread
        self.logger_thread = threading.Thread(target=self._logging_loop)
        self.logger_thread.daemon = True
        self.logger_thread.start()

        logger.info("Security audit logging started")

    def stop(self):
        """Stop audit logging"""
        logger = logging.getLogger(__name__)
        logger.info("Stopping security audit logging...")

        # Log system shutdown
        self.log_event(
            SecurityEventType.SYSTEM_SHUTDOWN,
            SecurityLevel.MEDIUM,
            "Security audit logging stopped",
            {}
        )

        self.running = False

        if self.logger_thread:
            self.logger_thread.join(timeout=5)

        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.end_session(session_id)

        logger.info("Security audit logging stopped")

    def start_session(self, user_id: Optional[str] = None, process_id: Optional[int] = None) -> str:
        """Start an audit session"""
        session_id = f"session_{int(time.time())}_{self.session_counter}"
        self.session_counter += 1

        if process_id is None:
            process_id = os.getpid()

        session = AuditSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            user_id=user_id,
            process_id=process_id,
            events_count=0,
            last_activity=time.time()
        )

        self.active_sessions[session_id] = session

        self.log_event(
            SecurityEventType.ME_SESSION_ESTABLISHED,
            SecurityLevel.LOW,
            f"Audit session started",
            {"session_id": session_id, "user_id": user_id, "process_id": process_id}
        )

        return session_id

    def end_session(self, session_id: str):
        """End an audit session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            session_duration = session.end_time - session.start_time

            self.log_event(
                SecurityEventType.ME_SESSION_CLOSED,
                SecurityLevel.LOW,
                f"Audit session ended",
                {
                    "session_id": session_id,
                    "duration_seconds": session_duration,
                    "events_count": session.events_count
                }
            )

            del self.active_sessions[session_id]

    def log_event(self, event_type: SecurityEventType, security_level: SecurityLevel,
                  description: str, details: Dict[str, Any], user_id: Optional[str] = None,
                  session_id: Optional[str] = None, outcome: str = "SUCCESS"):
        """Log a security event"""
        event_id = str(uuid.uuid4())
        timestamp = time.time()
        process_id = os.getpid()

        # Create event
        event = SecurityEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            security_level=security_level,
            source="military-tpm2",
            user_id=user_id,
            process_id=process_id,
            session_id=session_id,
            description=description,
            details=details,
            outcome=outcome,
            classification="UNCLASSIFIED // FOR OFFICIAL USE ONLY",
            checksum=""
        )

        # Calculate checksum
        event.checksum = self._calculate_event_checksum(event)

        # Update session activity
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.events_count += 1
            session.last_activity = timestamp

        # Queue event for logging
        with self.queue_lock:
            self.event_queue.append(event)

    def log_token_validation(self, token_id: str, validation_result: bool,
                           details: Dict[str, Any], session_id: Optional[str] = None):
        """Log military token validation event"""
        security_level = SecurityLevel.HIGH if validation_result else SecurityLevel.CRITICAL
        outcome = "SUCCESS" if validation_result else "FAILURE"
        description = f"Military token {token_id} validation {'succeeded' if validation_result else 'failed'}"

        self.log_event(
            SecurityEventType.TOKEN_VALIDATION,
            security_level,
            description,
            {"token_id": token_id, "validation_result": validation_result, **details},
            session_id=session_id,
            outcome=outcome
        )

    def log_authorization_decision(self, granted: bool, resource: str, operation: str,
                                 details: Dict[str, Any], user_id: Optional[str] = None,
                                 session_id: Optional[str] = None):
        """Log authorization decision"""
        if granted:
            event_type = SecurityEventType.AUTHORIZATION_GRANTED
            security_level = SecurityLevel.MEDIUM
            outcome = "SUCCESS"
            description = f"Authorization granted for {operation} on {resource}"
        else:
            event_type = SecurityEventType.AUTHORIZATION_DENIED
            security_level = SecurityLevel.HIGH
            outcome = "FAILURE"
            description = f"Authorization denied for {operation} on {resource}"

        self.log_event(
            event_type,
            security_level,
            description,
            {"resource": resource, "operation": operation, **details},
            user_id=user_id,
            session_id=session_id,
            outcome=outcome
        )

    def log_tpm_command(self, command: str, success: bool, details: Dict[str, Any],
                       session_id: Optional[str] = None):
        """Log TPM command processing"""
        security_level = SecurityLevel.LOW if success else SecurityLevel.MEDIUM
        outcome = "SUCCESS" if success else "FAILURE"
        description = f"TPM command {command} {'processed successfully' if success else 'failed'}"

        self.log_event(
            SecurityEventType.TPM_COMMAND_PROCESSED,
            security_level,
            description,
            {"command": command, "success": success, **details},
            session_id=session_id,
            outcome=outcome
        )

    def log_acceleration_usage(self, acceleration_type: str, operation: str,
                              performance_metrics: Dict[str, float],
                              session_id: Optional[str] = None):
        """Log hardware acceleration usage"""
        description = f"Hardware acceleration used: {acceleration_type} for {operation}"

        self.log_event(
            SecurityEventType.ACCELERATION_USAGE,
            SecurityLevel.LOW,
            description,
            {
                "acceleration_type": acceleration_type,
                "operation": operation,
                "performance_metrics": performance_metrics
            },
            session_id=session_id
        )

    def log_fallback_activation(self, from_acceleration: str, to_acceleration: str,
                               reason: str, session_id: Optional[str] = None):
        """Log fallback activation"""
        description = f"Fallback activated: {from_acceleration} -> {to_acceleration}"

        self.log_event(
            SecurityEventType.FALLBACK_ACTIVATION,
            SecurityLevel.MEDIUM,
            description,
            {
                "from_acceleration": from_acceleration,
                "to_acceleration": to_acceleration,
                "reason": reason
            },
            session_id=session_id
        )

    def log_security_violation(self, violation_type: str, severity: str,
                              details: Dict[str, Any], user_id: Optional[str] = None,
                              session_id: Optional[str] = None):
        """Log security violation"""
        security_level = SecurityLevel.CRITICAL
        description = f"Security violation detected: {violation_type}"

        self.log_event(
            SecurityEventType.SECURITY_VIOLATION,
            security_level,
            description,
            {"violation_type": violation_type, "severity": severity, **details},
            user_id=user_id,
            session_id=session_id,
            outcome="FAILURE"
        )

    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period"""
        end_time = time.time()
        start_time = end_time - (hours * 3600)

        summary = {
            "period_hours": hours,
            "start_time": start_time,
            "end_time": end_time,
            "events_by_type": {},
            "events_by_level": {},
            "events_by_outcome": {},
            "active_sessions": len(self.active_sessions),
            "total_events": 0,
            "security_violations": 0,
            "authorization_denials": 0
        }

        # Initialize counters
        for event_type in SecurityEventType:
            summary["events_by_type"][event_type.value] = 0

        for security_level in SecurityLevel:
            summary["events_by_level"][security_level.value] = 0

        for outcome in ["SUCCESS", "FAILURE", "ERROR"]:
            summary["events_by_outcome"][outcome] = 0

        # This would typically read from audit log files
        # For now, we'll use queued events as an example
        with self.queue_lock:
            for event in self.event_queue:
                if event.timestamp >= start_time:
                    summary["events_by_type"][event.event_type.value] += 1
                    summary["events_by_level"][event.security_level.value] += 1
                    summary["events_by_outcome"][event.outcome] += 1
                    summary["total_events"] += 1

                    if event.event_type == SecurityEventType.SECURITY_VIOLATION:
                        summary["security_violations"] += 1
                    elif event.event_type == SecurityEventType.AUTHORIZATION_DENIED:
                        summary["authorization_denials"] += 1

        return summary

    def export_audit_report(self, hours: int = 24, output_path: Optional[str] = None) -> str:
        """Export comprehensive audit report"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"/var/log/military-tpm/audit_report_{timestamp}.json"

        summary = self.get_audit_summary(hours)

        report = {
            "report_timestamp": time.time(),
            "report_id": str(uuid.uuid4()),
            "classification": "UNCLASSIFIED // FOR OFFICIAL USE ONLY",
            "summary": summary,
            "configuration": self.config,
            "active_sessions": {
                session_id: {
                    "start_time": session.start_time,
                    "user_id": session.user_id,
                    "process_id": session.process_id,
                    "events_count": session.events_count,
                    "last_activity": session.last_activity
                }
                for session_id, session in self.active_sessions.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger = logging.getLogger(__name__)
        logger.info(f"Audit report exported: {output_path}")
        return output_path

    def _load_config(self) -> Dict[str, Any]:
        """Load audit configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            # Use default configuration
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default audit configuration"""
        return {
            "enabled": True,
            "log_file": "/var/log/military-tpm/audit.log",
            "log_level": "INFO",
            "log_format": "structured",
            "events": {
                "token_validation": True,
                "authorization_decisions": True,
                "me_communication": True,
                "tpm_command_processing": True,
                "acceleration_usage": True,
                "fallback_activation": True,
                "security_violations": True
            },
            "rotation": {
                "max_size_mb": 100,
                "max_files": 10,
                "compress": True
            }
        }

    def _setup_logging(self):
        """Setup structured logging"""
        log_file = self.config.get("log_file", "/var/log/military-tpm/audit.log")
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create rotating file handler
        max_size_bytes = self.config.get("rotation", {}).get("max_size_mb", 100) * 1024 * 1024
        backup_count = self.config.get("rotation", {}).get("max_files", 10)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        # Setup audit logger
        audit_logger = logging.getLogger("military_tpm_audit")
        audit_logger.setLevel(log_level)
        audit_logger.addHandler(file_handler)
        audit_logger.propagate = False

        self.audit_logger = audit_logger

    def _initialize_audit_log(self):
        """Initialize audit log with system information"""
        self.audit_logger.info("=== AUDIT LOG INITIALIZED ===")
        self.audit_logger.info(f"Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY")
        self.audit_logger.info(f"System: {os.uname().sysname} {os.uname().release}")
        self.audit_logger.info(f"Configuration: {self.config_path}")

    def _logging_loop(self):
        """Main logging loop"""
        while self.running:
            try:
                events_to_log = []

                # Get events from queue
                with self.queue_lock:
                    if self.event_queue:
                        events_to_log = self.event_queue.copy()
                        self.event_queue.clear()

                # Log events
                for event in events_to_log:
                    self._log_event_to_file(event)

                time.sleep(0.1)  # Short sleep to prevent busy loop

            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error in logging loop: {e}")
                time.sleep(1)

    def _log_event_to_file(self, event: SecurityEvent):
        """Log event to audit file"""
        # Create structured log entry
        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(event.timestamp)),
            "sequence": self.sequence_number,
            "event_type": event.event_type.value,
            "security_level": event.security_level.value,
            "source": event.source,
            "user_id": event.user_id,
            "process_id": event.process_id,
            "session_id": event.session_id,
            "description": event.description,
            "details": event.details,
            "outcome": event.outcome,
            "classification": event.classification,
            "checksum": event.checksum
        }

        self.sequence_number += 1

        # Log as structured JSON
        self.audit_logger.info(json.dumps(log_entry))

    def _calculate_event_checksum(self, event: SecurityEvent) -> str:
        """Calculate tamper-evident checksum for event"""
        # Create data string for checksum
        data_parts = [
            event.event_id,
            str(event.timestamp),
            event.event_type.value,
            str(event.security_level.value),
            event.source,
            event.user_id or "",
            str(event.process_id),
            event.session_id or "",
            event.description,
            json.dumps(event.details, sort_keys=True),
            event.outcome,
            event.classification
        ]

        data_string = "|".join(data_parts)
        return hashlib.sha256(data_string.encode()).hexdigest()


def main():
    """Main audit logger entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="TPM2 Security Audit Logger")
    parser.add_argument("--config", default="/etc/military-tpm/audit.json",
                       help="Configuration file path")
    parser.add_argument("--export-report", type=int, metavar="HOURS",
                       help="Export audit report for specified hours and exit")
    parser.add_argument("--summary", type=int, metavar="HOURS",
                       help="Show audit summary for specified hours and exit")

    args = parser.parse_args()

    # Create audit logger
    audit_logger = SecurityAuditLogger(args.config)

    if args.export_report:
        # Export report and exit
        audit_logger.start()
        time.sleep(1)  # Allow initialization
        report_path = audit_logger.export_audit_report(args.export_report)
        print(f"Audit report exported: {report_path}")
        audit_logger.stop()
        return

    if args.summary:
        # Show summary and exit
        audit_logger.start()
        time.sleep(1)  # Allow initialization
        summary = audit_logger.get_audit_summary(args.summary)
        print(json.dumps(summary, indent=2))
        audit_logger.stop()
        return

    # Run audit logger
    def signal_handler(signum, frame):
        logger = logging.getLogger(__name__)
        logger.info("Received shutdown signal")
        audit_logger.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        audit_logger.start()

        # Keep running until signal received
        while audit_logger.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Keyboard interrupt received")
    finally:
        audit_logger.stop()


if __name__ == "__main__":
    main()