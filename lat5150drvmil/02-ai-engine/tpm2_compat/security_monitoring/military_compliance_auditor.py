#!/usr/bin/env python3
"""
Military Compliance Audit System with Tamper-Evident Trails
Comprehensive audit system for military compliance standards

Author: Military Compliance Audit Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import hashlib
import hmac
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import base64
import zipfile
import tempfile
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MilitaryStandard(Enum):
    """Military compliance standards"""
    FIPS_140_2_LEVEL_1 = "fips_140_2_level_1"
    FIPS_140_2_LEVEL_2 = "fips_140_2_level_2"
    FIPS_140_2_LEVEL_3 = "fips_140_2_level_3"
    FIPS_140_2_LEVEL_4 = "fips_140_2_level_4"
    COMMON_CRITERIA_EAL1 = "common_criteria_eal1"
    COMMON_CRITERIA_EAL2 = "common_criteria_eal2"
    COMMON_CRITERIA_EAL3 = "common_criteria_eal3"
    COMMON_CRITERIA_EAL4 = "common_criteria_eal4"
    COMMON_CRITERIA_EAL5 = "common_criteria_eal5"
    COMMON_CRITERIA_EAL6 = "common_criteria_eal6"
    COMMON_CRITERIA_EAL7 = "common_criteria_eal7"
    STIG_CAT_I = "stig_category_i"
    STIG_CAT_II = "stig_category_ii"
    STIG_CAT_III = "stig_category_iii"
    NIST_800_53_LOW = "nist_800_53_low"
    NIST_800_53_MODERATE = "nist_800_53_moderate"
    NIST_800_53_HIGH = "nist_800_53_high"
    DOD_8500_2 = "dod_8500_2"
    ISO_27001 = "iso_27001"
    FISMA = "fisma"
    DIACAP = "diacap"

class ComplianceStatus(Enum):
    """Compliance status levels"""
    UNKNOWN = "unknown"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    REMEDIATION_REQUIRED = "remediation_required"
    UNDER_REVIEW = "under_review"

class AuditType(Enum):
    """Audit types"""
    AUTOMATED = "automated"
    MANUAL = "manual"
    HYBRID = "hybrid"
    CONTINUOUS = "continuous"
    SPOT_CHECK = "spot_check"
    COMPREHENSIVE = "comprehensive"

class EvidenceType(Enum):
    """Evidence types"""
    CONFIGURATION = "configuration"
    LOG_DATA = "log_data"
    SYSTEM_OUTPUT = "system_output"
    DOCUMENTATION = "documentation"
    SCREENSHOT = "screenshot"
    CERTIFICATE = "certificate"
    TEST_RESULT = "test_result"
    VULNERABILITY_SCAN = "vulnerability_scan"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    standard: MilitaryStandard
    category: str
    title: str
    description: str
    control_families: List[str]
    security_functions: List[str]
    assessment_procedures: List[str]
    evidence_requirements: List[EvidenceType]
    automated_checks: List[str]
    manual_procedures: List[str]
    priority: int
    risk_level: str

@dataclass
class AuditEvidence:
    """Audit evidence record"""
    evidence_id: str
    timestamp: float
    requirement_id: str
    evidence_type: EvidenceType
    description: str
    collector: str
    data_source: str
    raw_data: Optional[str]
    processed_data: Optional[Dict[str, Any]]
    file_path: Optional[str]
    checksum: str
    digital_signature: Optional[str]
    chain_of_custody: List[Dict[str, Any]]
    tamper_evident: bool

@dataclass
class ComplianceAssessment:
    """Compliance assessment result"""
    assessment_id: str
    timestamp: float
    requirement_id: str
    auditor: str
    audit_type: AuditType
    status: ComplianceStatus
    score: float
    findings: List[str]
    deficiencies: List[str]
    recommendations: List[str]
    evidence_references: List[str]
    risk_assessment: Dict[str, Any]
    remediation_plan: Optional[Dict[str, Any]]
    next_review_date: Optional[float]

@dataclass
class AuditTrail:
    """Tamper-evident audit trail entry"""
    entry_id: str
    timestamp: float
    previous_hash: str
    entry_hash: str
    entry_type: str
    actor: str
    action: str
    target: str
    details: Dict[str, Any]
    integrity_signature: str

class MilitaryComplianceAuditor:
    """
    Military compliance audit system with tamper-evident trails
    Provides comprehensive compliance monitoring and audit capabilities
    """

    def __init__(self, config_path: str = "/etc/military-tpm/compliance_audit.json"):
        """Initialize military compliance auditor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False

        # Database for persistent storage
        self.db_path = self.config.get("database_path", "/var/lib/military-tpm/compliance_audit.db")
        self._init_database()

        # Cryptographic keys for digital signatures
        self.private_key, self.public_key = self._load_or_generate_keys()

        # Compliance requirements
        self.requirements = self._load_compliance_requirements()
        self.assessment_procedures = self._load_assessment_procedures()

        # Audit state
        self.audit_trail = []
        self.active_assessments = {}
        self.evidence_store = {}
        self.chain_of_custody = {}

        # Monitoring threads
        self.monitoring_threads = []
        self.last_trail_hash = self._get_last_trail_hash()

        # Initialize compliance baseline
        self._initialize_compliance_baseline()

        logger.info("Military Compliance Auditor initialized")

    def start(self):
        """Start compliance auditing"""
        logger.info("Starting military compliance auditing...")
        self.running = True

        # Start monitoring threads
        self._start_continuous_monitoring()
        self._start_automated_assessments()
        self._start_evidence_collection()
        self._start_trail_integrity_monitor()

        # Log audit start
        self._add_audit_trail_entry(
            "SYSTEM",
            "START_AUDIT",
            "compliance_auditor",
            {"version": "1.0", "standards": [s.value for s in self._get_enabled_standards()]}
        )

        logger.info("Military compliance auditing started")

    def stop(self):
        """Stop compliance auditing"""
        logger.info("Stopping military compliance auditing...")
        self.running = False

        # Stop all monitoring threads
        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        # Log audit stop
        self._add_audit_trail_entry(
            "SYSTEM",
            "STOP_AUDIT",
            "compliance_auditor",
            {"duration_seconds": time.time() - self._get_audit_start_time()}
        )

        logger.info("Military compliance auditing stopped")

    def perform_compliance_assessment(self, standard: MilitaryStandard,
                                    auditor: str, audit_type: AuditType = AuditType.AUTOMATED) -> str:
        """Perform comprehensive compliance assessment"""
        assessment_id = str(uuid.uuid4())
        timestamp = time.time()

        # Get requirements for standard
        requirements = self._get_requirements_for_standard(standard)

        # Log assessment start
        self._add_audit_trail_entry(
            auditor,
            "START_ASSESSMENT",
            f"standard:{standard.value}",
            {"assessment_id": assessment_id, "requirements_count": len(requirements)}
        )

        assessment_results = []

        for requirement in requirements:
            try:
                # Collect evidence
                evidence = self._collect_requirement_evidence(requirement, auditor)

                # Perform assessment
                result = self._assess_requirement_compliance(
                    requirement, evidence, auditor, audit_type
                )

                assessment_results.append(result)

                # Store assessment result
                self._store_assessment_result(result)

                # Update audit trail
                self._add_audit_trail_entry(
                    auditor,
                    "ASSESS_REQUIREMENT",
                    requirement.requirement_id,
                    {
                        "status": result.status.value,
                        "score": result.score,
                        "evidence_count": len(result.evidence_references)
                    }
                )

            except Exception as e:
                logger.error(f"Error assessing requirement {requirement.requirement_id}: {e}")
                # Log assessment error
                self._add_audit_trail_entry(
                    auditor,
                    "ASSESSMENT_ERROR",
                    requirement.requirement_id,
                    {"error": str(e)}
                )

        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(
            assessment_id, standard, assessment_results, auditor, audit_type
        )

        # Log assessment completion
        self._add_audit_trail_entry(
            auditor,
            "COMPLETE_ASSESSMENT",
            f"standard:{standard.value}",
            {
                "assessment_id": assessment_id,
                "overall_status": overall_assessment["status"],
                "compliance_score": overall_assessment["score"]
            }
        )

        return assessment_id

    def collect_audit_evidence(self, requirement_id: str, evidence_type: EvidenceType,
                             description: str, collector: str, data_source: str,
                             raw_data: Optional[str] = None,
                             file_path: Optional[str] = None) -> str:
        """Collect and store audit evidence with tamper protection"""
        evidence_id = str(uuid.uuid4())
        timestamp = time.time()

        # Calculate checksum
        checksum_data = f"{evidence_id}|{timestamp}|{requirement_id}|{evidence_type.value}|{description}"
        if raw_data:
            checksum_data += f"|{raw_data}"
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                checksum_data += f"|{hashlib.sha256(f.read()).hexdigest()}"

        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()

        # Create digital signature
        digital_signature = self._create_digital_signature(checksum_data)

        # Initialize chain of custody
        chain_of_custody = [{
            "timestamp": timestamp,
            "actor": collector,
            "action": "COLLECT",
            "location": data_source,
            "integrity_hash": checksum
        }]

        # Create evidence record
        evidence = AuditEvidence(
            evidence_id=evidence_id,
            timestamp=timestamp,
            requirement_id=requirement_id,
            evidence_type=evidence_type,
            description=description,
            collector=collector,
            data_source=data_source,
            raw_data=raw_data,
            processed_data=None,
            file_path=file_path,
            checksum=checksum,
            digital_signature=digital_signature,
            chain_of_custody=chain_of_custody,
            tamper_evident=True
        )

        # Store evidence
        self._store_evidence(evidence)

        # Update audit trail
        self._add_audit_trail_entry(
            collector,
            "COLLECT_EVIDENCE",
            requirement_id,
            {
                "evidence_id": evidence_id,
                "evidence_type": evidence_type.value,
                "data_source": data_source
            }
        )

        logger.info(f"Evidence collected: {evidence_id} for requirement {requirement_id}")
        return evidence_id

    def verify_audit_trail_integrity(self) -> Dict[str, Any]:
        """Verify integrity of complete audit trail"""
        logger.info("Verifying audit trail integrity...")

        verification_results = {
            "timestamp": time.time(),
            "total_entries": 0,
            "verified_entries": 0,
            "failed_entries": 0,
            "integrity_violations": [],
            "hash_chain_valid": True,
            "digital_signatures_valid": True,
            "overall_status": "UNKNOWN"
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entry_id, timestamp, previous_hash, entry_hash, entry_type,
                       actor, action, target, details, integrity_signature
                FROM audit_trail ORDER BY timestamp
            """)

            entries = cursor.fetchall()
            verification_results["total_entries"] = len(entries)

            previous_hash = ""
            for entry in entries:
                entry_id, timestamp, prev_hash, entry_hash, entry_type, actor, action, target, details_json, signature = entry

                try:
                    # Verify hash chain
                    if prev_hash != previous_hash:
                        verification_results["hash_chain_valid"] = False
                        verification_results["integrity_violations"].append({
                            "entry_id": entry_id,
                            "violation_type": "HASH_CHAIN_BREAK",
                            "expected_previous_hash": previous_hash,
                            "actual_previous_hash": prev_hash
                        })

                    # Verify entry hash
                    calculated_hash = self._calculate_entry_hash(
                        entry_id, timestamp, prev_hash, entry_type, actor, action, target, details_json
                    )
                    if calculated_hash != entry_hash:
                        verification_results["integrity_violations"].append({
                            "entry_id": entry_id,
                            "violation_type": "ENTRY_HASH_MISMATCH",
                            "calculated_hash": calculated_hash,
                            "stored_hash": entry_hash
                        })

                    # Verify digital signature
                    if not self._verify_digital_signature(calculated_hash, signature):
                        verification_results["digital_signatures_valid"] = False
                        verification_results["integrity_violations"].append({
                            "entry_id": entry_id,
                            "violation_type": "INVALID_SIGNATURE"
                        })

                    if not verification_results["integrity_violations"] or \
                       not any(v["entry_id"] == entry_id for v in verification_results["integrity_violations"]):
                        verification_results["verified_entries"] += 1
                    else:
                        verification_results["failed_entries"] += 1

                    previous_hash = entry_hash

                except Exception as e:
                    logger.error(f"Error verifying entry {entry_id}: {e}")
                    verification_results["failed_entries"] += 1
                    verification_results["integrity_violations"].append({
                        "entry_id": entry_id,
                        "violation_type": "VERIFICATION_ERROR",
                        "error": str(e)
                    })

        # Determine overall status
        if not verification_results["integrity_violations"]:
            verification_results["overall_status"] = "VERIFIED"
        elif verification_results["verified_entries"] > verification_results["failed_entries"]:
            verification_results["overall_status"] = "PARTIALLY_VERIFIED"
        else:
            verification_results["overall_status"] = "COMPROMISED"

        # Log verification results
        self._add_audit_trail_entry(
            "SYSTEM",
            "VERIFY_TRAIL_INTEGRITY",
            "audit_trail",
            verification_results
        )

        logger.info(f"Audit trail integrity verification completed: {verification_results['overall_status']}")
        return verification_results

    def export_compliance_package(self, assessment_id: str, include_evidence: bool = True) -> str:
        """Export complete compliance package with all evidence and audit trails"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        package_path = f"/var/lib/military-tpm/compliance_packages/package_{assessment_id}_{timestamp}.zip"

        os.makedirs(os.path.dirname(package_path), exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create package structure
            package_dir = Path(temp_dir) / "compliance_package"
            package_dir.mkdir(exist_ok=True)

            # Export assessment data
            assessment_data = self._get_assessment_data(assessment_id)
            with open(package_dir / "assessment.json", 'w') as f:
                json.dump(assessment_data, f, indent=2)

            # Export audit trail
            audit_trail_data = self._get_audit_trail_data(assessment_id)
            with open(package_dir / "audit_trail.json", 'w') as f:
                json.dump(audit_trail_data, f, indent=2)

            # Export evidence if requested
            if include_evidence:
                evidence_dir = package_dir / "evidence"
                evidence_dir.mkdir(exist_ok=True)
                self._export_evidence_files(assessment_id, evidence_dir)

            # Create integrity manifest
            manifest = self._create_package_manifest(package_dir)
            with open(package_dir / "MANIFEST.json", 'w') as f:
                json.dump(manifest, f, indent=2)

            # Create digital signature for package
            package_signature = self._create_package_signature(manifest)
            with open(package_dir / "SIGNATURE.txt", 'w') as f:
                f.write(package_signature)

            # Create ZIP package
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)

        # Log package export
        self._add_audit_trail_entry(
            "SYSTEM",
            "EXPORT_COMPLIANCE_PACKAGE",
            assessment_id,
            {
                "package_path": package_path,
                "include_evidence": include_evidence,
                "package_size_bytes": os.path.getsize(package_path)
            }
        )

        logger.info(f"Compliance package exported: {package_path}")
        return package_path

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive compliance dashboard data"""
        dashboard_data = {
            "timestamp": time.time(),
            "overall_compliance": self._calculate_overall_compliance(),
            "standards_status": self._get_standards_compliance_status(),
            "recent_assessments": self._get_recent_assessments(30),
            "compliance_trends": self._get_compliance_trends(90),
            "deficiency_summary": self._get_deficiency_summary(),
            "remediation_status": self._get_remediation_status(),
            "audit_trail_health": self._get_audit_trail_health(),
            "evidence_collection_status": self._get_evidence_collection_status(),
            "upcoming_reviews": self._get_upcoming_reviews(),
            "risk_assessment": self._get_compliance_risk_assessment()
        }

        return dashboard_data

    def _load_config(self) -> Dict[str, Any]:
        """Load compliance audit configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default compliance audit configuration"""
        return {
            "enabled": True,
            "database_path": "/var/lib/military-tpm/compliance_audit.db",
            "keys_directory": "/etc/military-tpm/audit_keys",
            "evidence_storage": "/var/lib/military-tpm/evidence",
            "enabled_standards": [
                "fips_140_2_level_2",
                "common_criteria_eal4",
                "stig_category_ii",
                "nist_800_53_moderate"
            ],
            "audit_settings": {
                "continuous_monitoring": True,
                "automated_assessment_interval_hours": 24,
                "evidence_retention_days": 2555,  # 7 years
                "trail_integrity_check_interval_hours": 6,
                "digital_signatures_required": True,
                "tamper_detection_enabled": True
            },
            "assessment_thresholds": {
                "compliance_score_pass": 80.0,
                "critical_deficiency_threshold": 1,
                "high_risk_threshold": 3,
                "evidence_collection_timeout_hours": 48
            },
            "remediation": {
                "automated_remediation": False,
                "escalation_timeout_hours": 72,
                "notification_channels": ["email", "syslog"]
            }
        }

    def _init_database(self):
        """Initialize compliance audit database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Compliance requirements table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_requirements (
                    requirement_id TEXT PRIMARY KEY,
                    standard TEXT,
                    category TEXT,
                    title TEXT,
                    description TEXT,
                    control_families TEXT,
                    security_functions TEXT,
                    assessment_procedures TEXT,
                    evidence_requirements TEXT,
                    automated_checks TEXT,
                    manual_procedures TEXT,
                    priority INTEGER,
                    risk_level TEXT
                )
            """)

            # Assessment results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assessment_results (
                    assessment_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    requirement_id TEXT,
                    auditor TEXT,
                    audit_type TEXT,
                    status TEXT,
                    score REAL,
                    findings TEXT,
                    deficiencies TEXT,
                    recommendations TEXT,
                    evidence_references TEXT,
                    risk_assessment TEXT,
                    remediation_plan TEXT,
                    next_review_date REAL,
                    FOREIGN KEY(requirement_id) REFERENCES compliance_requirements(requirement_id)
                )
            """)

            # Evidence table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    requirement_id TEXT,
                    evidence_type TEXT,
                    description TEXT,
                    collector TEXT,
                    data_source TEXT,
                    raw_data TEXT,
                    processed_data TEXT,
                    file_path TEXT,
                    checksum TEXT,
                    digital_signature TEXT,
                    chain_of_custody TEXT,
                    tamper_evident BOOLEAN,
                    FOREIGN KEY(requirement_id) REFERENCES compliance_requirements(requirement_id)
                )
            """)

            # Audit trail table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    entry_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    previous_hash TEXT,
                    entry_hash TEXT,
                    entry_type TEXT,
                    actor TEXT,
                    action TEXT,
                    target TEXT,
                    details TEXT,
                    integrity_signature TEXT
                )
            """)

            # Compliance baselines table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_baselines (
                    baseline_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    standard TEXT,
                    baseline_data TEXT,
                    checksum TEXT
                )
            """)

            conn.commit()

    # Additional implementation methods would continue here...
    # This includes methods for:
    # - Cryptographic key management
    # - Compliance requirement loading
    # - Assessment procedures
    # - Evidence collection algorithms
    # - Digital signature creation/verification
    # - Audit trail management
    # - etc.


def main():
    """Main compliance auditor entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Military Compliance Auditor")
    parser.add_argument("--config", default="/etc/military-tpm/compliance_audit.json",
                       help="Configuration file path")
    parser.add_argument("--assess", choices=[s.value for s in MilitaryStandard],
                       help="Perform compliance assessment for standard")
    parser.add_argument("--verify-trail", action="store_true",
                       help="Verify audit trail integrity and exit")
    parser.add_argument("--dashboard", action="store_true",
                       help="Show compliance dashboard and exit")
    parser.add_argument("--export-package", metavar="ASSESSMENT_ID",
                       help="Export compliance package for assessment")

    args = parser.parse_args()

    # Create compliance auditor
    auditor = MilitaryComplianceAuditor(args.config)

    if args.assess:
        # Perform assessment and exit
        auditor.start()
        time.sleep(2)  # Allow initialization
        standard = MilitaryStandard(args.assess)
        assessment_id = auditor.perform_compliance_assessment(standard, "cli_user")
        print(f"Assessment completed: {assessment_id}")
        auditor.stop()
        return

    if args.verify_trail:
        # Verify trail and exit
        auditor.start()
        time.sleep(1)  # Allow initialization
        results = auditor.verify_audit_trail_integrity()
        print(json.dumps(results, indent=2))
        auditor.stop()
        return

    if args.dashboard:
        # Show dashboard and exit
        auditor.start()
        time.sleep(2)  # Allow data collection
        dashboard = auditor.get_compliance_dashboard()
        print(json.dumps(dashboard, indent=2))
        auditor.stop()
        return

    if args.export_package:
        # Export package and exit
        auditor.start()
        time.sleep(1)  # Allow initialization
        package_path = auditor.export_compliance_package(args.export_package)
        print(f"Compliance package exported: {package_path}")
        auditor.stop()
        return

    # Run compliance auditor
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        auditor.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        auditor.start()

        # Keep running until signal received
        while auditor.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        auditor.stop()


if __name__ == "__main__":
    main()