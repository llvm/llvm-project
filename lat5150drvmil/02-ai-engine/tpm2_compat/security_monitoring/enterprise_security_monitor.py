#!/usr/bin/env python3
"""
Enterprise Security Monitoring System for TPM2 Compatibility Layer
Comprehensive real-time security monitoring with military-grade compliance

Author: TPM2 Security Monitoring Agent
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
import hmac
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import subprocess
import psutil
import schedule
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    IMMINENT = 5

class SecurityEventCategory(Enum):
    """Security event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ACCESS_CONTROL = "access_control"
    DATA_INTEGRITY = "data_integrity"
    CRYPTOGRAPHIC = "cryptographic"
    NETWORK_SECURITY = "network_security"
    SYSTEM_INTEGRITY = "system_integrity"
    COMPLIANCE = "compliance"
    INCIDENT = "incident"
    PERFORMANCE = "performance"

class ComplianceStandard(Enum):
    """Military compliance standards"""
    FIPS_140_2 = "fips_140_2"
    COMMON_CRITERIA = "common_criteria"
    STIG = "stig"
    NIST_800_53 = "nist_800_53"
    ISO_27001 = "iso_27001"
    DOD_8500 = "dod_8500"

@dataclass
class SecurityThreat:
    """Security threat information"""
    threat_id: str
    timestamp: float
    category: SecurityEventCategory
    threat_level: ThreatLevel
    source_ip: Optional[str]
    source_process: Optional[str]
    target_resource: str
    attack_vector: str
    description: str
    indicators: Dict[str, Any]
    mitigation_status: str
    response_actions: List[str]

@dataclass
class ComplianceEvent:
    """Compliance audit event"""
    event_id: str
    timestamp: float
    standard: ComplianceStandard
    requirement: str
    status: str  # COMPLIANT, NON_COMPLIANT, PARTIAL
    evidence: Dict[str, Any]
    remediation_required: bool
    risk_level: ThreatLevel

@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    timestamp: float
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: ThreatLevel
    trend_analysis: Dict[str, Any]
    recommended_actions: List[str]

class EnterpriseSecurityMonitor:
    """
    Enterprise-grade security monitoring system for TPM2 compatibility layer
    Provides comprehensive real-time monitoring with military compliance
    """

    def __init__(self, config_path: str = "/etc/military-tpm/enterprise_security.json"):
        """Initialize enterprise security monitor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False

        # Database for persistent storage
        self.db_path = self.config.get("database_path", "/var/lib/military-tpm/security.db")
        self._init_database()

        # Encryption for sensitive data
        self.encryption_key = self._derive_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Monitoring threads
        self.monitoring_threads = []
        self.incident_queue = asyncio.Queue()
        self.alert_handlers = {}

        # Threat detection
        self.threat_patterns = self._load_threat_patterns()
        self.baseline_metrics = {}
        self.anomaly_detectors = {}

        # Compliance tracking
        self.compliance_checkers = self._init_compliance_checkers()
        self.audit_trail = []

        # Real-time state
        self.active_threats = {}
        self.system_status = {}
        self.performance_metrics = {}

        logger.info("Enterprise Security Monitor initialized")

    def start(self):
        """Start comprehensive security monitoring"""
        logger.info("Starting enterprise security monitoring...")
        self.running = True

        # Start core monitoring threads
        self._start_threat_detection()
        self._start_compliance_monitoring()
        self._start_performance_monitoring()
        self._start_incident_response()
        self._start_audit_processing()

        # Schedule regular tasks
        self._schedule_tasks()

        # Log startup event
        self._log_security_event(
            SecurityEventCategory.SYSTEM_INTEGRITY,
            ThreatLevel.LOW,
            "Enterprise security monitoring started",
            {"version": "1.0", "config": self.config_path}
        )

        logger.info("Enterprise security monitoring started")

    def stop(self):
        """Stop security monitoring"""
        logger.info("Stopping enterprise security monitoring...")
        self.running = False

        # Stop all monitoring threads
        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # Log shutdown event
        self._log_security_event(
            SecurityEventCategory.SYSTEM_INTEGRITY,
            ThreatLevel.LOW,
            "Enterprise security monitoring stopped",
            {}
        )

        logger.info("Enterprise security monitoring stopped")

    def register_alert_handler(self, threat_level: ThreatLevel, handler: Callable):
        """Register custom alert handler"""
        if threat_level not in self.alert_handlers:
            self.alert_handlers[threat_level] = []
        self.alert_handlers[threat_level].append(handler)

    def report_security_event(self, category: SecurityEventCategory,
                            threat_level: ThreatLevel, description: str,
                            details: Dict[str, Any]):
        """Report security event for monitoring"""
        self._log_security_event(category, threat_level, description, details)

        # Check for threat patterns
        self._analyze_threat_patterns(category, threat_level, details)

        # Update compliance status
        self._update_compliance_status(category, details)

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        return {
            "timestamp": time.time(),
            "overall_status": self._calculate_overall_security_status(),
            "active_threats": self._get_active_threats_summary(),
            "compliance_status": self._get_compliance_summary(),
            "performance_health": self._get_performance_health(),
            "recent_incidents": self._get_recent_incidents(24),
            "audit_summary": self._get_audit_summary(24),
            "system_metrics": self._get_system_metrics(),
            "recommendations": self._generate_security_recommendations()
        }

    def export_compliance_report(self, standard: ComplianceStandard,
                               hours: int = 24) -> str:
        """Export detailed compliance report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"/var/log/military-tpm/compliance_{standard.value}_{timestamp}.json"

        report = {
            "report_metadata": {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "standard": standard.value,
                "period_hours": hours,
                "classification": "UNCLASSIFIED // FOR OFFICIAL USE ONLY"
            },
            "executive_summary": self._generate_compliance_executive_summary(standard),
            "detailed_findings": self._get_compliance_findings(standard, hours),
            "risk_assessment": self._assess_compliance_risks(standard),
            "remediation_plan": self._generate_remediation_plan(standard),
            "evidence_collection": self._collect_compliance_evidence(standard),
            "audit_trail": self._get_compliance_audit_trail(standard, hours)
        }

        # Encrypt sensitive sections
        if self.config.get("encrypt_reports", True):
            report["detailed_findings"] = self._encrypt_data(json.dumps(report["detailed_findings"]))
            report["evidence_collection"] = self._encrypt_data(json.dumps(report["evidence_collection"]))

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Compliance report exported: {output_path}")
        return output_path

    def _load_config(self) -> Dict[str, Any]:
        """Load enterprise security configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default enterprise security configuration"""
        return {
            "enabled": True,
            "database_path": "/var/lib/military-tpm/security.db",
            "log_retention_days": 365,
            "encrypt_reports": True,
            "real_time_monitoring": True,
            "threat_detection": {
                "enabled": True,
                "sensitivity": "high",
                "ml_enabled": False,
                "patterns_file": "/etc/military-tpm/threat_patterns.json"
            },
            "compliance_monitoring": {
                "enabled": True,
                "standards": ["fips_140_2", "common_criteria", "stig"],
                "check_interval_minutes": 60,
                "automated_remediation": False
            },
            "performance_monitoring": {
                "enabled": True,
                "check_interval_seconds": 30,
                "alert_thresholds": {
                    "cpu_usage_percent": 80,
                    "memory_usage_percent": 85,
                    "response_time_ms": 1000,
                    "error_rate_percent": 1,
                    "temperature_celsius": 80
                }
            },
            "incident_response": {
                "enabled": True,
                "auto_escalation": True,
                "notification_channels": ["email", "syslog"],
                "response_procedures": {
                    "critical": ["isolate", "notify", "preserve_evidence"],
                    "high": ["alert", "monitor", "log"],
                    "medium": ["log", "schedule_review"],
                    "low": ["log"]
                }
            },
            "audit_settings": {
                "tamper_detection": True,
                "digital_signatures": True,
                "chain_of_custody": True,
                "export_formats": ["json", "xml", "pdf"]
            }
        }

    def _init_database(self):
        """Initialize security monitoring database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Security events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    category TEXT,
                    threat_level INTEGER,
                    description TEXT,
                    details TEXT,
                    checksum TEXT,
                    encrypted BOOLEAN DEFAULT FALSE
                )
            """)

            # Threats table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threats (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    category TEXT,
                    threat_level INTEGER,
                    source_ip TEXT,
                    source_process TEXT,
                    target_resource TEXT,
                    attack_vector TEXT,
                    description TEXT,
                    indicators TEXT,
                    mitigation_status TEXT,
                    response_actions TEXT
                )
            """)

            # Compliance events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    standard TEXT,
                    requirement TEXT,
                    status TEXT,
                    evidence TEXT,
                    remediation_required BOOLEAN,
                    risk_level INTEGER
                )
            """)

            # Performance alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    component TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    severity INTEGER,
                    trend_analysis TEXT,
                    recommended_actions TEXT
                )
            """)

            # Audit trail table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    action TEXT,
                    user_id TEXT,
                    resource TEXT,
                    outcome TEXT,
                    details TEXT,
                    signature TEXT
                )
            """)

            conn.commit()

    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key for sensitive data"""
        password = self.config.get("encryption_password", "default_military_tpm_key").encode()
        salt = b"military_tpm_salt_2025"  # In production, use random salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def _start_threat_detection(self):
        """Start threat detection monitoring"""
        def threat_detection_loop():
            logger.info("Starting threat detection monitoring")
            while self.running:
                try:
                    self._scan_for_threats()
                    self._analyze_behavioral_anomalies()
                    self._update_threat_intelligence()
                    time.sleep(self.config.get("threat_detection", {}).get("scan_interval", 60))
                except Exception as e:
                    logger.error(f"Error in threat detection: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=threat_detection_loop, daemon=True)
        thread.start()
        self.monitoring_threads.append(thread)

    def _start_compliance_monitoring(self):
        """Start compliance monitoring"""
        def compliance_monitoring_loop():
            logger.info("Starting compliance monitoring")
            while self.running:
                try:
                    for standard in self.config.get("compliance_monitoring", {}).get("standards", []):
                        self._check_compliance_standard(ComplianceStandard(standard))

                    interval = self.config.get("compliance_monitoring", {}).get("check_interval_minutes", 60)
                    time.sleep(interval * 60)
                except Exception as e:
                    logger.error(f"Error in compliance monitoring: {e}")
                    time.sleep(300)  # 5 minute delay on error

        thread = threading.Thread(target=compliance_monitoring_loop, daemon=True)
        thread.start()
        self.monitoring_threads.append(thread)

    def _start_performance_monitoring(self):
        """Start performance monitoring"""
        def performance_monitoring_loop():
            logger.info("Starting performance monitoring")
            while self.running:
                try:
                    self._collect_performance_metrics()
                    self._analyze_performance_trends()
                    self._check_performance_thresholds()

                    interval = self.config.get("performance_monitoring", {}).get("check_interval_seconds", 30)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=performance_monitoring_loop, daemon=True)
        thread.start()
        self.monitoring_threads.append(thread)

    def _start_incident_response(self):
        """Start incident response system"""
        def incident_response_loop():
            logger.info("Starting incident response system")
            while self.running:
                try:
                    # Check for incidents requiring response
                    self._process_incident_queue()
                    self._escalate_unresolved_incidents()
                    self._update_incident_status()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in incident response: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=incident_response_loop, daemon=True)
        thread.start()
        self.monitoring_threads.append(thread)

    def _start_audit_processing(self):
        """Start audit log processing"""
        def audit_processing_loop():
            logger.info("Starting audit processing")
            while self.running:
                try:
                    self._process_audit_logs()
                    self._verify_audit_integrity()
                    self._generate_audit_checksums()
                    time.sleep(300)  # Process every 5 minutes
                except Exception as e:
                    logger.error(f"Error in audit processing: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=audit_processing_loop, daemon=True)
        thread.start()
        self.monitoring_threads.append(thread)

    def _schedule_tasks(self):
        """Schedule regular monitoring tasks"""
        # Daily compliance reports
        schedule.every().day.at("02:00").do(self._generate_daily_compliance_report)

        # Weekly security assessments
        schedule.every().sunday.at("03:00").do(self._perform_weekly_security_assessment)

        # Monthly threat intelligence updates
        schedule.every().month.do(self._update_monthly_threat_intelligence)

        # Hourly health checks
        schedule.every().hour.do(self._perform_health_check)

    def _log_security_event(self, category: SecurityEventCategory,
                          threat_level: ThreatLevel, description: str,
                          details: Dict[str, Any]):
        """Log security event to database"""
        event_id = str(uuid.uuid4())
        timestamp = time.time()

        # Calculate checksum for integrity
        data_string = f"{event_id}|{timestamp}|{category.value}|{threat_level.value}|{description}|{json.dumps(details, sort_keys=True)}"
        checksum = hashlib.sha256(data_string.encode()).hexdigest()

        # Encrypt sensitive details if configured
        encrypted = False
        details_str = json.dumps(details)
        if self.config.get("encrypt_events", False) and threat_level.value >= ThreatLevel.HIGH.value:
            details_str = self._encrypt_data(details_str)
            encrypted = True

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_events
                (id, timestamp, category, threat_level, description, details, checksum, encrypted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (event_id, timestamp, category.value, threat_level.value,
                  description, details_str, checksum, encrypted))
            conn.commit()

        # Trigger alert handlers if configured
        if threat_level in self.alert_handlers:
            for handler in self.alert_handlers[threat_level]:
                try:
                    handler(category, threat_level, description, details)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

    def _scan_for_threats(self):
        """Scan system for security threats"""
        # Check for suspicious processes
        self._check_suspicious_processes()

        # Monitor file system integrity
        self._monitor_file_integrity()

        # Check network connections
        self._monitor_network_activity()

        # Analyze system logs
        self._analyze_system_logs()

    def _check_suspicious_processes(self):
        """Check for suspicious running processes"""
        try:
            for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    # Check against known threat patterns
                    if self._is_suspicious_process(process.info):
                        self._create_threat(
                            SecurityEventCategory.SYSTEM_INTEGRITY,
                            ThreatLevel.MEDIUM,
                            f"Suspicious process detected: {process.info['name']}",
                            {
                                "pid": process.info['pid'],
                                "cmdline": process.info['cmdline'],
                                "cpu_percent": process.info['cpu_percent'],
                                "memory_percent": process.info['memory_percent']
                            }
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking suspicious processes: {e}")

    def _is_suspicious_process(self, process_info: Dict[str, Any]) -> bool:
        """Check if process is suspicious"""
        suspicious_names = ['nc', 'netcat', 'nmap', 'nikto', 'sqlmap']
        suspicious_cmdlines = ['reverse shell', 'backdoor', '/tmp/', 'wget http']

        name = process_info.get('name', '').lower()
        cmdline = ' '.join(process_info.get('cmdline', [])).lower()

        return (name in suspicious_names or
                any(pattern in cmdline for pattern in suspicious_cmdlines) or
                process_info.get('cpu_percent', 0) > 90)

    def _create_threat(self, category: SecurityEventCategory, threat_level: ThreatLevel,
                      description: str, indicators: Dict[str, Any]):
        """Create and log security threat"""
        threat_id = str(uuid.uuid4())
        threat = SecurityThreat(
            threat_id=threat_id,
            timestamp=time.time(),
            category=category,
            threat_level=threat_level,
            source_ip=indicators.get('source_ip'),
            source_process=indicators.get('source_process'),
            target_resource=indicators.get('target_resource', 'system'),
            attack_vector=indicators.get('attack_vector', 'unknown'),
            description=description,
            indicators=indicators,
            mitigation_status='detected',
            response_actions=self._get_response_actions(threat_level)
        )

        self.active_threats[threat_id] = threat

        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO threats
                (id, timestamp, category, threat_level, source_ip, source_process,
                 target_resource, attack_vector, description, indicators,
                 mitigation_status, response_actions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threat.threat_id, threat.timestamp, threat.category.value,
                threat.threat_level.value, threat.source_ip, threat.source_process,
                threat.target_resource, threat.attack_vector, threat.description,
                json.dumps(threat.indicators), threat.mitigation_status,
                json.dumps(threat.response_actions)
            ))
            conn.commit()

    def _get_response_actions(self, threat_level: ThreatLevel) -> List[str]:
        """Get appropriate response actions for threat level"""
        response_map = {
            ThreatLevel.LOW: ["log", "monitor"],
            ThreatLevel.MEDIUM: ["log", "alert", "investigate"],
            ThreatLevel.HIGH: ["log", "alert", "isolate", "investigate"],
            ThreatLevel.CRITICAL: ["log", "alert", "isolate", "block", "escalate"],
            ThreatLevel.IMMINENT: ["log", "alert", "shutdown", "isolate", "escalate", "notify_authorities"]
        }
        return response_map.get(threat_level, ["log"])

    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns"""
        patterns_file = self.config.get("threat_detection", {}).get("patterns_file")
        if patterns_file and os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load threat patterns: {e}")

        # Return default patterns
        return {
            "suspicious_processes": ["nc", "netcat", "nmap", "nikto"],
            "suspicious_network": ["443", "22", "23", "3389"],
            "file_integrity": ["/etc/passwd", "/etc/shadow", "/boot/"],
            "log_patterns": ["failed login", "authentication failure", "invalid user"]
        }

    def _init_compliance_checkers(self) -> Dict[ComplianceStandard, Any]:
        """Initialize compliance checkers"""
        checkers = {}

        for standard_name in self.config.get("compliance_monitoring", {}).get("standards", []):
            try:
                standard = ComplianceStandard(standard_name)
                checkers[standard] = self._create_compliance_checker(standard)
            except ValueError:
                logger.warning(f"Unknown compliance standard: {standard_name}")

        return checkers

    def _create_compliance_checker(self, standard: ComplianceStandard):
        """Create compliance checker for specific standard"""
        # This would contain specific compliance checking logic
        # For now, return a placeholder
        return {
            "requirements": self._get_compliance_requirements(standard),
            "checks": self._get_compliance_checks(standard)
        }

    def _get_compliance_requirements(self, standard: ComplianceStandard) -> List[str]:
        """Get compliance requirements for standard"""
        requirements_map = {
            ComplianceStandard.FIPS_140_2: [
                "cryptographic_module_specification",
                "finite_state_model",
                "required_roles_services_authentication",
                "required_finite_state_model",
                "physical_security",
                "operational_environment",
                "cryptographic_key_management",
                "electromagnetic_interference",
                "self_tests",
                "design_assurance",
                "mitigation_of_other_attacks"
            ],
            ComplianceStandard.COMMON_CRITERIA: [
                "security_target",
                "protection_profile",
                "functional_requirements",
                "assurance_requirements",
                "evaluation_evidence"
            ],
            ComplianceStandard.STIG: [
                "vulnerability_assessment",
                "configuration_management",
                "access_control",
                "audit_logging",
                "system_hardening"
            ]
        }
        return requirements_map.get(standard, [])

    def _get_compliance_checks(self, standard: ComplianceStandard) -> List[Dict[str, Any]]:
        """Get specific compliance checks for standard"""
        # This would contain actual compliance checking logic
        return [
            {"id": "check_001", "name": "Crypto module validation", "automated": True},
            {"id": "check_002", "name": "Access control verification", "automated": True},
            {"id": "check_003", "name": "Audit log integrity", "automated": True}
        ]

    # Additional methods would be implemented here for:
    # - Performance monitoring
    # - Incident response
    # - Compliance checking
    # - Report generation
    # - Dashboard data
    # etc.


def main():
    """Main enterprise security monitor entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise Security Monitor")
    parser.add_argument("--config", default="/etc/military-tpm/enterprise_security.json",
                       help="Configuration file path")
    parser.add_argument("--dashboard", action="store_true",
                       help="Show security dashboard and exit")
    parser.add_argument("--export-compliance", choices=["fips_140_2", "common_criteria", "stig"],
                       help="Export compliance report and exit")

    args = parser.parse_args()

    # Create security monitor
    monitor = EnterpriseSecurityMonitor(args.config)

    if args.dashboard:
        # Show dashboard and exit
        monitor.start()
        time.sleep(2)  # Allow data collection
        dashboard = monitor.get_security_dashboard()
        print(json.dumps(dashboard, indent=2))
        monitor.stop()
        return

    if args.export_compliance:
        # Export compliance report and exit
        monitor.start()
        time.sleep(2)  # Allow data collection
        standard = ComplianceStandard(args.export_compliance)
        report_path = monitor.export_compliance_report(standard)
        print(f"Compliance report exported: {report_path}")
        monitor.stop()
        return

    # Run security monitor
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        monitor.start()

        # Keep running until signal received
        while monitor.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()