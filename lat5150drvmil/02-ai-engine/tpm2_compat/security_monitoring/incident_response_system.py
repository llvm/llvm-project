#!/usr/bin/env python3
"""
Automated Security Incident Response System
Comprehensive incident detection, classification, and automated response

Author: Security Incident Response Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import asyncio
import smtplib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import uuid
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import psutil
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    EMERGENCY = 6

class IncidentCategory(Enum):
    """Incident categories"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTION = "malware_detection"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    SYSTEM_COMPROMISE = "system_compromise"
    CONFIGURATION_VIOLATION = "configuration_violation"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    HARDWARE_FAILURE = "hardware_failure"
    COMPLIANCE_VIOLATION = "compliance_violation"
    INSIDER_THREAT = "insider_threat"
    ADVANCED_PERSISTENT_THREAT = "advanced_persistent_threat"

class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    BLOCK_USER = "block_user"
    ISOLATE_SYSTEM = "isolate_system"
    KILL_PROCESS = "kill_process"
    BACKUP_DATA = "backup_data"
    PRESERVE_EVIDENCE = "preserve_evidence"
    ESCALATE = "escalate"
    NOTIFY_ADMIN = "notify_admin"
    SHUTDOWN_SERVICE = "shutdown_service"
    ENABLE_MONITORING = "enable_monitoring"
    RESET_CREDENTIALS = "reset_credentials"
    QUARANTINE_FILE = "quarantine_file"

class IncidentStatus(Enum):
    """Incident status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    timestamp: float
    severity: IncidentSeverity
    category: IncidentCategory
    title: str
    description: str
    source_system: str
    affected_assets: List[str]
    indicators: Dict[str, Any]
    evidence: List[str]
    initial_response: List[ResponseAction]
    status: IncidentStatus
    assigned_to: Optional[str]
    escalation_level: int
    containment_actions: List[str]
    recovery_actions: List[str]
    lessons_learned: List[str]

@dataclass
class ResponseRule:
    """Incident response rule"""
    rule_id: str
    name: str
    category: IncidentCategory
    severity_threshold: IncidentSeverity
    conditions: Dict[str, Any]
    automated_actions: List[ResponseAction]
    manual_actions: List[str]
    notification_channels: List[str]
    escalation_timeout: int
    enabled: bool

@dataclass
class EscalationPolicy:
    """Incident escalation policy"""
    policy_id: str
    name: str
    severity_levels: List[IncidentSeverity]
    escalation_chain: List[Dict[str, Any]]
    timeout_minutes: int
    notification_methods: List[str]
    business_hours_only: bool
    enabled: bool

class IncidentResponseSystem:
    """
    Automated security incident response system
    Provides comprehensive incident detection, classification, and response
    """

    def __init__(self, config_path: str = "/etc/military-tpm/incident_response.json"):
        """Initialize incident response system"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False

        # Database for persistent storage
        self.db_path = self.config.get("database_path", "/var/lib/military-tpm/incident_response.db")
        self._init_database()

        # Response rules and policies
        self.response_rules = self._load_response_rules()
        self.escalation_policies = self._load_escalation_policies()

        # Active incidents and response state
        self.active_incidents = {}
        self.incident_history = deque(maxlen=10000)
        self.response_queue = asyncio.Queue()
        self.escalation_timers = {}

        # Monitoring and detection
        self.detection_engines = {}
        self.threat_indicators = defaultdict(int)
        self.behavioral_baselines = {}

        # Response capabilities
        self.response_handlers = {}
        self.notification_clients = {}
        self.forensic_collectors = {}

        # Statistics and metrics
        self.response_metrics = {
            "incidents_detected": 0,
            "incidents_resolved": 0,
            "false_positives": 0,
            "average_response_time": 0.0,
            "escalations_triggered": 0
        }

        # Initialize response system
        self._initialize_response_handlers()
        self._initialize_notification_clients()
        self._load_threat_intelligence()

        logger.info("Incident Response System initialized")

    def start(self):
        """Start incident response system"""
        logger.info("Starting incident response system...")
        self.running = True

        # Start core response threads
        self._start_incident_detection()
        self._start_response_processing()
        self._start_escalation_monitoring()
        self._start_threat_hunting()
        self._start_metrics_collection()

        logger.info("Incident response system started")

    def stop(self):
        """Stop incident response system"""
        logger.info("Stopping incident response system...")
        self.running = False

        # Process any remaining incidents
        self._process_remaining_incidents()

        logger.info("Incident response system stopped")

    def report_security_event(self, category: IncidentCategory, severity: IncidentSeverity,
                            title: str, description: str, source_system: str,
                            indicators: Dict[str, Any], evidence: List[str] = None) -> str:
        """Report security event for incident processing"""
        incident_id = str(uuid.uuid4())
        timestamp = time.time()

        # Create incident record
        incident = SecurityIncident(
            incident_id=incident_id,
            timestamp=timestamp,
            severity=severity,
            category=category,
            title=title,
            description=description,
            source_system=source_system,
            affected_assets=indicators.get("affected_assets", []),
            indicators=indicators,
            evidence=evidence or [],
            initial_response=[],
            status=IncidentStatus.DETECTED,
            assigned_to=None,
            escalation_level=0,
            containment_actions=[],
            recovery_actions=[],
            lessons_learned=[]
        )

        # Store incident
        self.active_incidents[incident_id] = incident
        self.incident_history.append(incident)
        self._store_incident_in_db(incident)

        # Trigger immediate response
        self._trigger_incident_response(incident)

        # Update metrics
        self.response_metrics["incidents_detected"] += 1

        logger.info(f"Security incident reported: {incident_id} - {title}")
        return incident_id

    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific incident"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            return {
                "incident_id": incident_id,
                "status": incident.status.value,
                "severity": incident.severity.value,
                "category": incident.category.value,
                "title": incident.title,
                "timestamp": incident.timestamp,
                "assigned_to": incident.assigned_to,
                "escalation_level": incident.escalation_level,
                "containment_actions": incident.containment_actions,
                "recovery_actions": incident.recovery_actions
            }
        return None

    def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents"""
        active_incidents = []
        for incident in self.active_incidents.values():
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE]:
                active_incidents.append({
                    "incident_id": incident.incident_id,
                    "severity": incident.severity.value,
                    "category": incident.category.value,
                    "title": incident.title,
                    "timestamp": incident.timestamp,
                    "status": incident.status.value,
                    "escalation_level": incident.escalation_level
                })
        return sorted(active_incidents, key=lambda x: (x["severity"], x["timestamp"]), reverse=True)

    def update_incident_status(self, incident_id: str, status: IncidentStatus,
                             assigned_to: Optional[str] = None,
                             notes: Optional[str] = None) -> bool:
        """Update incident status"""
        if incident_id not in self.active_incidents:
            return False

        incident = self.active_incidents[incident_id]
        old_status = incident.status
        incident.status = status

        if assigned_to:
            incident.assigned_to = assigned_to

        # Log status change
        self._log_incident_update(incident_id, f"Status changed from {old_status.value} to {status.value}", notes)

        # Update database
        self._update_incident_in_db(incident)

        # Handle status-specific actions
        if status == IncidentStatus.RESOLVED:
            self._handle_incident_resolution(incident)
        elif status == IncidentStatus.CLOSED:
            self._handle_incident_closure(incident)
        elif status == IncidentStatus.FALSE_POSITIVE:
            self._handle_false_positive(incident)

        return True

    def escalate_incident(self, incident_id: str, escalation_reason: str) -> bool:
        """Escalate incident to next level"""
        if incident_id not in self.active_incidents:
            return False

        incident = self.active_incidents[incident_id]
        incident.escalation_level += 1

        # Find appropriate escalation policy
        escalation_policy = self._get_escalation_policy(incident.severity, incident.category)
        if escalation_policy and incident.escalation_level < len(escalation_policy.escalation_chain):
            escalation_target = escalation_policy.escalation_chain[incident.escalation_level]

            # Execute escalation
            self._execute_escalation(incident, escalation_target, escalation_reason)

            # Log escalation
            self._log_incident_update(incident_id, f"Escalated to level {incident.escalation_level}", escalation_reason)

            # Update metrics
            self.response_metrics["escalations_triggered"] += 1

            return True

        return False

    def get_response_dashboard(self) -> Dict[str, Any]:
        """Get incident response dashboard data"""
        current_time = time.time()

        # Active incidents summary
        active_by_severity = defaultdict(int)
        active_by_category = defaultdict(int)
        for incident in self.active_incidents.values():
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE]:
                active_by_severity[incident.severity.value] += 1
                active_by_category[incident.category.value] += 1

        # Recent incidents (last 24 hours)
        recent_incidents = [
            incident for incident in self.incident_history
            if current_time - incident.timestamp < 86400
        ]

        # Response time analysis
        resolved_incidents = [
            incident for incident in self.incident_history
            if incident.status == IncidentStatus.RESOLVED and hasattr(incident, 'resolution_time')
        ]

        avg_response_time = 0.0
        if resolved_incidents:
            total_response_time = sum(getattr(incident, 'resolution_time', 0) for incident in resolved_incidents)
            avg_response_time = total_response_time / len(resolved_incidents)

        dashboard = {
            "timestamp": current_time,
            "active_incidents": {
                "total": len([i for i in self.active_incidents.values()
                            if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE]]),
                "by_severity": dict(active_by_severity),
                "by_category": dict(active_by_category)
            },
            "recent_activity": {
                "incidents_last_24h": len(recent_incidents),
                "resolved_last_24h": len([i for i in recent_incidents if i.status == IncidentStatus.RESOLVED]),
                "false_positives_last_24h": len([i for i in recent_incidents if i.status == IncidentStatus.FALSE_POSITIVE])
            },
            "performance_metrics": {
                "average_response_time_minutes": avg_response_time / 60,
                "incidents_detected": self.response_metrics["incidents_detected"],
                "incidents_resolved": self.response_metrics["incidents_resolved"],
                "false_positive_rate": (self.response_metrics["false_positives"] /
                                      max(self.response_metrics["incidents_detected"], 1)) * 100,
                "escalations_triggered": self.response_metrics["escalations_triggered"]
            },
            "system_health": {
                "response_queue_size": self.response_queue.qsize() if hasattr(self.response_queue, 'qsize') else 0,
                "active_rules": len([r for r in self.response_rules.values() if r.enabled]),
                "detection_engines": len(self.detection_engines),
                "notification_clients": len(self.notification_clients)
            },
            "top_threat_indicators": dict(sorted(self.threat_indicators.items(),
                                               key=lambda x: x[1], reverse=True)[:10])
        }

        return dashboard

    def _load_config(self) -> Dict[str, Any]:
        """Load incident response configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default incident response configuration"""
        return {
            "enabled": True,
            "database_path": "/var/lib/military-tpm/incident_response.db",
            "response_settings": {
                "auto_response_enabled": True,
                "escalation_enabled": True,
                "forensic_collection": True,
                "threat_hunting": True,
                "behavioral_analysis": True
            },
            "notification": {
                "email": {
                    "enabled": True,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "security@military-tpm.local"
                },
                "syslog": {
                    "enabled": True,
                    "server": "localhost",
                    "port": 514,
                    "facility": "local0"
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            },
            "response_timeouts": {
                "initial_response_seconds": 60,
                "containment_timeout_minutes": 30,
                "escalation_timeout_minutes": 120,
                "resolution_timeout_hours": 24
            },
            "automated_actions": {
                "critical_incidents": ["isolate_system", "preserve_evidence", "escalate"],
                "high_incidents": ["block_ip", "alert", "preserve_evidence"],
                "medium_incidents": ["alert", "log_only"],
                "low_incidents": ["log_only"]
            },
            "threat_intelligence": {
                "feeds_enabled": True,
                "update_interval_hours": 6,
                "sources": ["internal", "public"]
            }
        }

    def _init_database(self):
        """Initialize incident response database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Incidents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    severity INTEGER,
                    category TEXT,
                    title TEXT,
                    description TEXT,
                    source_system TEXT,
                    affected_assets TEXT,
                    indicators TEXT,
                    evidence TEXT,
                    initial_response TEXT,
                    status TEXT,
                    assigned_to TEXT,
                    escalation_level INTEGER,
                    containment_actions TEXT,
                    recovery_actions TEXT,
                    lessons_learned TEXT
                )
            """)

            # Response rules table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    severity_threshold INTEGER,
                    conditions TEXT,
                    automated_actions TEXT,
                    manual_actions TEXT,
                    notification_channels TEXT,
                    escalation_timeout INTEGER,
                    enabled BOOLEAN
                )
            """)

            # Escalation policies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS escalation_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT,
                    severity_levels TEXT,
                    escalation_chain TEXT,
                    timeout_minutes INTEGER,
                    notification_methods TEXT,
                    business_hours_only BOOLEAN,
                    enabled BOOLEAN
                )
            """)

            # Response actions log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_actions (
                    action_id TEXT PRIMARY KEY,
                    incident_id TEXT,
                    timestamp REAL,
                    action_type TEXT,
                    action_details TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    FOREIGN KEY(incident_id) REFERENCES incidents(incident_id)
                )
            """)

            conn.commit()

    def _trigger_incident_response(self, incident: SecurityIncident):
        """Trigger automated incident response"""
        try:
            # Find matching response rules
            matching_rules = self._find_matching_rules(incident)

            # Execute automated actions
            for rule in matching_rules:
                if rule.enabled:
                    self._execute_automated_actions(incident, rule.automated_actions)

            # Schedule escalation if configured
            if self.config.get("response_settings", {}).get("escalation_enabled", True):
                self._schedule_escalation(incident)

            # Start containment procedures for high-severity incidents
            if incident.severity.value >= IncidentSeverity.HIGH.value:
                self._initiate_containment(incident)

            # Begin forensic collection if enabled
            if self.config.get("response_settings", {}).get("forensic_collection", True):
                self._start_forensic_collection(incident)

        except Exception as e:
            logger.error(f"Error triggering response for incident {incident.incident_id}: {e}")

    def _find_matching_rules(self, incident: SecurityIncident) -> List[ResponseRule]:
        """Find response rules that match the incident"""
        matching_rules = []

        for rule in self.response_rules.values():
            if (rule.category == incident.category and
                incident.severity.value >= rule.severity_threshold.value and
                self._evaluate_rule_conditions(incident, rule.conditions)):
                matching_rules.append(rule)

        return matching_rules

    def _execute_automated_actions(self, incident: SecurityIncident, actions: List[ResponseAction]):
        """Execute automated response actions"""
        for action in actions:
            try:
                success = False
                error_message = None

                if action == ResponseAction.LOG_ONLY:
                    logger.warning(f"INCIDENT: {incident.title} - {incident.description}")
                    success = True

                elif action == ResponseAction.ALERT:
                    self._send_alert(incident)
                    success = True

                elif action == ResponseAction.BLOCK_IP:
                    success = self._block_ip_address(incident.indicators.get("source_ip"))

                elif action == ResponseAction.BLOCK_USER:
                    success = self._block_user_account(incident.indicators.get("user_id"))

                elif action == ResponseAction.ISOLATE_SYSTEM:
                    success = self._isolate_system(incident.affected_assets)

                elif action == ResponseAction.KILL_PROCESS:
                    success = self._kill_malicious_process(incident.indicators.get("process_id"))

                elif action == ResponseAction.PRESERVE_EVIDENCE:
                    success = self._preserve_evidence(incident)

                elif action == ResponseAction.ESCALATE:
                    success = self.escalate_incident(incident.incident_id, "Automated escalation")

                # Log action result
                self._log_response_action(incident.incident_id, action, success, error_message)

                # Update incident response list
                incident.initial_response.append(action)

            except Exception as e:
                logger.error(f"Error executing action {action.value} for incident {incident.incident_id}: {e}")
                self._log_response_action(incident.incident_id, action, False, str(e))

    # Additional implementation methods would continue here...
    # This includes methods for:
    # - Notification systems
    # - Escalation management
    # - Forensic collection
    # - Threat hunting
    # - Response action execution
    # - etc.


def main():
    """Main incident response system entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Incident Response System")
    parser.add_argument("--config", default="/etc/military-tpm/incident_response.json",
                       help="Configuration file path")
    parser.add_argument("--dashboard", action="store_true",
                       help="Show response dashboard and exit")
    parser.add_argument("--list-incidents", action="store_true",
                       help="List active incidents and exit")
    parser.add_argument("--test-alert", action="store_true",
                       help="Send test alert and exit")

    args = parser.parse_args()

    # Create incident response system
    irs = IncidentResponseSystem(args.config)

    if args.dashboard:
        # Show dashboard and exit
        irs.start()
        time.sleep(2)  # Allow initialization
        dashboard = irs.get_response_dashboard()
        print(json.dumps(dashboard, indent=2))
        irs.stop()
        return

    if args.list_incidents:
        # List incidents and exit
        irs.start()
        time.sleep(1)  # Allow initialization
        incidents = irs.get_active_incidents()
        print(json.dumps(incidents, indent=2))
        irs.stop()
        return

    if args.test_alert:
        # Send test alert and exit
        irs.start()
        time.sleep(1)  # Allow initialization
        incident_id = irs.report_security_event(
            IncidentCategory.SYSTEM_COMPROMISE,
            IncidentSeverity.MEDIUM,
            "Test Security Alert",
            "This is a test security alert to verify incident response system functionality",
            "test_system",
            {"test": True}
        )
        print(f"Test incident created: {incident_id}")
        time.sleep(5)  # Allow processing
        irs.stop()
        return

    # Run incident response system
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        irs.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        irs.start()

        # Keep running until signal received
        while irs.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        irs.stop()


if __name__ == "__main__":
    main()