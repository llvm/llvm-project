#!/usr/bin/env python3
"""
Device 0x8014: Certificate Store

Secure storage and management of X.509 certificates, certificate chains,
and certificate revocation lists (CRLs) for PKI operations.

Device ID: 0x8014
Group: 1 (Extended Security)
Risk Level: SAFE (READ operations are safe)

POST-QUANTUM CRYPTOGRAPHY COMPLIANCE:
- Digital Signatures: ML-DSA-87 (FIPS 204) for all new certificates
- Hybrid Certificates: Supports both classical + PQC signatures
- Certificate Storage: AES-256-GCM encrypted (FIPS 197 + SP 800-38D)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)

✓ All new certificates use ML-DSA-87 signatures
✓ Legacy RSA/ECDSA certificates marked for migration
✓ Hybrid mode supports dual classical + PQC signatures

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 2.0.0 (Post-Quantum)
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from pqc_constants import (
    MLDSAAlgorithm,
    AESAlgorithm,
    HashAlgorithm,
    REQUIRED_SIGNATURE,
    REQUIRED_HASH,
    is_pqc_compliant,
)
from typing import Dict, List, Optional, Any


class CertificateType(object):
    """Certificate types"""
    ROOT_CA = 0
    INTERMEDIATE_CA = 1
    SERVER_CERT = 2
    CLIENT_CERT = 3
    CODE_SIGNING = 4
    EMAIL = 5


class CertificateStoreDevice(DSMILDeviceBase):
    """Certificate Store (0x8014)"""

    # Register map
    REG_STORE_STATUS = 0x00
    REG_CERT_COUNT = 0x04
    REG_CAPACITY = 0x08
    REG_CRL_COUNT = 0x0C

    # Status bits
    STATUS_STORE_READY = 0x01
    STATUS_ENCRYPTED = 0x02
    STATUS_CRL_VALID = 0x04

    def __init__(self, device_id: int = 0x8014,
                 name: str = "Certificate Store",
                 description: str = "PKI Certificate Storage"):
        super().__init__(device_id, name, description)

        # Device-specific state (PQC-compliant certificates)
        self.certificates = {
            1: {
                "type": CertificateType.ROOT_CA,
                "subject": "CN=Root CA",
                "valid": True,
                "signature_algorithm": "ML-DSA-87",
                "hash_algorithm": "SHA-512",
                "pqc_compliant": True,
                "hybrid": False,
            },
            2: {
                "type": CertificateType.SERVER_CERT,
                "subject": "CN=server.mil",
                "valid": True,
                "signature_algorithm": "ML-DSA-87",
                "hash_algorithm": "SHA-512",
                "pqc_compliant": True,
                "hybrid": False,
            },
            3: {
                "type": CertificateType.CLIENT_CERT,
                "subject": "CN=client.mil",
                "valid": True,
                "signature_algorithm": "Hybrid (RSA-4096 + ML-DSA-87)",
                "hash_algorithm": "SHA-512",
                "pqc_compliant": True,
                "hybrid": True,
                "migration_note": "Transitional hybrid certificate",
            },
            4: {
                "type": CertificateType.SERVER_CERT,
                "subject": "CN=legacy.mil",
                "valid": True,
                "signature_algorithm": "RSA-2048",
                "hash_algorithm": "SHA-256",
                "pqc_compliant": False,
                "hybrid": False,
                "deprecation_notice": "Migrate to ML-DSA-87",
                "quantum_vulnerable": True,
            },
        }
        self.max_certificates = 512
        self.crl_count = 1

        # Register map
        self.register_map = {
            "STORE_STATUS": {
                "offset": self.REG_STORE_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Certificate store status"
            },
            "CERT_COUNT": {
                "offset": self.REG_CERT_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of stored certificates"
            },
            "CAPACITY": {
                "offset": self.REG_CAPACITY,
                "size": 4,
                "access": "RO",
                "description": "Maximum certificate capacity"
            },
            "CRL_COUNT": {
                "offset": self.REG_CRL_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of CRLs"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Certificate Store"""
        try:
            self.state = DeviceState.INITIALIZING
            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "certificates": len(self.certificates),
                "capacity": self.max_certificates,
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.ENCRYPTED_STORAGE,
            DeviceCapability.STATUS_REPORTING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        return {
            "ready": True,
            "encrypted": True,
            "crl_valid": True,
            "certificates": len(self.certificates),
            "available": self.max_certificates - len(self.certificates),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "STORE_STATUS":
                value = 0x07  # All status bits set
            elif register == "CERT_COUNT":
                value = len(self.certificates)
            elif register == "CAPACITY":
                value = self.max_certificates
            elif register == "CRL_COUNT":
                value = self.crl_count
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

    def list_certificates(self) -> OperationResult:
        """List stored certificates"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        certs = []
        for cert_id, cert_info in self.certificates.items():
            certs.append({
                "id": cert_id,
                "type": cert_info["type"],
                "subject": cert_info["subject"],
                "valid": cert_info["valid"],
                "signature_algorithm": cert_info.get("signature_algorithm", "Unknown"),
                "pqc_compliant": cert_info.get("pqc_compliant", False),
            })

        self._record_operation(True)
        return OperationResult(True, data={"certificates": certs})

    def check_pqc_compliance(self) -> OperationResult:
        """Check Post-Quantum Cryptography compliance for all certificates"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        pqc_certificates = []
        hybrid_certificates = []
        legacy_certificates = []

        for cert_id, cert_info in self.certificates.items():
            is_compliant = cert_info.get('pqc_compliant', False)
            is_hybrid = cert_info.get('hybrid', False)
            is_vulnerable = cert_info.get('quantum_vulnerable', False)

            cert_entry = {
                "id": cert_id,
                "subject": cert_info["subject"],
                "type": cert_info["type"],
                "signature_algorithm": cert_info.get("signature_algorithm", "Unknown"),
                "hash_algorithm": cert_info.get("hash_algorithm", "Unknown"),
                "valid": cert_info["valid"],
            }

            if is_vulnerable:
                cert_entry["warning"] = cert_info.get("deprecation_notice", "Quantum-vulnerable")
                legacy_certificates.append(cert_entry)
            elif is_hybrid:
                cert_entry["note"] = cert_info.get("migration_note", "Hybrid certificate")
                hybrid_certificates.append(cert_entry)
            elif is_compliant:
                pqc_certificates.append(cert_entry)
            else:
                cert_entry["warning"] = "Not PQC-compliant"
                legacy_certificates.append(cert_entry)

        total = len(self.certificates)
        compliance_report = {
            "overall_compliant": len(legacy_certificates) == 0,
            "pqc_certificates": pqc_certificates,
            "hybrid_certificates": hybrid_certificates,
            "legacy_certificates": legacy_certificates,
            "total_certificates": total,
            "pqc_percentage": f"{len(pqc_certificates)/total*100:.1f}%" if total else "0%",
            "hybrid_percentage": f"{len(hybrid_certificates)/total*100:.1f}%" if total else "0%",
            "required_signature": "ML-DSA-87",
            "required_hash": "SHA-512",
            "hybrid_support": True,
        }

        self._record_operation(True)
        return OperationResult(True, data=compliance_report)
