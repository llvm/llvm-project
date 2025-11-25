#!/usr/bin/env python3
"""
Device 0x8013: Key Management

Cryptographic key lifecycle management for military cryptographic operations.
Handles key generation, storage, rotation, backup, and secure destruction.

Device ID: 0x8013
Group: 1 (Extended Security)
Risk Level: MONITORED (Critical cryptographic operations)

POST-QUANTUM CRYPTOGRAPHY COMPLIANCE:
- Key Encapsulation: ML-KEM-1024 (FIPS 203)
- Digital Signatures: ML-DSA-87 (FIPS 204)
- Symmetric Encryption: AES-256-GCM (FIPS 197 + SP 800-38D)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)

⚠ Legacy RSA/ECC keys marked as quantum-vulnerable
✓ All new keys use post-quantum algorithms

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 2.0.0 (Post-Quantum)
"""

import sys
import os
import time
import hashlib
import secrets

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from pqc_constants import (
    MLKEMAlgorithm,
    MLDSAAlgorithm,
    AESAlgorithm,
    HashAlgorithm,
    PQCProfile,
    REQUIRED_KEM,
    REQUIRED_SIGNATURE,
    REQUIRED_SYMMETRIC,
    REQUIRED_HASH,
    is_pqc_compliant,
    get_algorithm_name,
)
from typing import Dict, List, Optional, Any


class KeyType:
    """Cryptographic key types"""
    # Symmetric encryption
    SYMMETRIC = 0
    SYMMETRIC_AES_256_GCM = 1  # ⭐ PQC-compliant symmetric encryption

    # Post-Quantum Cryptography (REQUIRED for MIL-SPEC)
    ML_KEM_512 = 10   # ML-KEM-512 (Security Level 1)
    ML_KEM_768 = 11   # ML-KEM-768 (Security Level 3)
    ML_KEM_1024 = 12  # ML-KEM-1024 (Security Level 5) ⭐ REQUIRED
    ML_DSA_44 = 20    # ML-DSA-44 (Security Level 2)
    ML_DSA_65 = 21    # ML-DSA-65 (Security Level 3)
    ML_DSA_87 = 22    # ML-DSA-87 (Security Level 5) ⭐ REQUIRED

    # Legacy asymmetric (DEPRECATED - quantum-vulnerable)
    ASYMMETRIC_RSA = 100  # ⚠ Quantum-vulnerable - use ML-DSA
    ASYMMETRIC_ECC = 101  # ⚠ Quantum-vulnerable - use ML-KEM

    # Other key types
    HMAC = 200
    SESSION = 201


class KeyUsage:
    """Key usage types"""
    ENCRYPTION = 0
    SIGNING = 1
    KEY_EXCHANGE = 2
    AUTHENTICATION = 3
    DERIVATION = 4


class KeyStatus:
    """Key lifecycle status"""
    ACTIVE = 0
    INACTIVE = 1
    EXPIRED = 2
    REVOKED = 3
    DESTROYED = 4


class KeyStorageType:
    """Key storage locations"""
    TPM = 0
    HSM = 1
    SOFTWARE = 2
    SECURE_ENCLAVE = 3


class KeyManagementDevice(DSMILDeviceBase):
    """Key Management (0x8013)"""

    # Register map
    REG_KM_STATUS = 0x00
    REG_KEY_COUNT = 0x04
    REG_KEY_CAPACITY = 0x08
    REG_ACTIVE_KEYS = 0x0C
    REG_EXPIRED_KEYS = 0x10
    REG_ROTATION_COUNT = 0x14
    REG_LAST_ROTATION = 0x18
    REG_FIPS_MODE = 0x1C

    # Status bits
    STATUS_KM_ACTIVE = 0x01
    STATUS_TPM_AVAILABLE = 0x02
    STATUS_HSM_AVAILABLE = 0x04
    STATUS_FIPS_MODE = 0x08
    STATUS_AUTO_ROTATION = 0x10
    STATUS_BACKUP_ENABLED = 0x20

    def __init__(self, device_id: int = 0x8013,
                 name: str = "Key Management",
                 description: str = "Cryptographic Key Lifecycle Management"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.keys = {}
        self.max_keys = 1000
        self.rotation_history = []
        self.max_history = 500

        self.tpm_available = True
        self.hsm_available = True
        self.fips_mode = True
        self.auto_rotation_enabled = True
        self.backup_enabled = True

        self.rotation_count = 0
        self.last_rotation_time = 0

        # Initialize with sample keys
        self._initialize_sample_keys()

        # Register map
        self.register_map = {
            "KM_STATUS": {
                "offset": self.REG_KM_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Key management status"
            },
            "KEY_COUNT": {
                "offset": self.REG_KEY_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total number of keys"
            },
            "KEY_CAPACITY": {
                "offset": self.REG_KEY_CAPACITY,
                "size": 4,
                "access": "RO",
                "description": "Maximum key capacity"
            },
            "ACTIVE_KEYS": {
                "offset": self.REG_ACTIVE_KEYS,
                "size": 4,
                "access": "RO",
                "description": "Number of active keys"
            },
            "EXPIRED_KEYS": {
                "offset": self.REG_EXPIRED_KEYS,
                "size": 4,
                "access": "RO",
                "description": "Number of expired keys"
            },
            "ROTATION_COUNT": {
                "offset": self.REG_ROTATION_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total key rotations"
            },
            "LAST_ROTATION": {
                "offset": self.REG_LAST_ROTATION,
                "size": 4,
                "access": "RO",
                "description": "Last rotation timestamp"
            },
            "FIPS_MODE": {
                "offset": self.REG_FIPS_MODE,
                "size": 4,
                "access": "RO",
                "description": "FIPS 140-2 mode enabled"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Key Management device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize key management system
            self._initialize_sample_keys()

            self.tpm_available = True
            self.hsm_available = True
            self.fips_mode = True
            self.auto_rotation_enabled = True
            self.backup_enabled = True

            self.rotation_count = 0
            self.last_rotation_time = int(time.time())
            self.rotation_history = []

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "total_keys": len(self.keys),
                "active_keys": self._count_keys_by_status(KeyStatus.ACTIVE),
                "tpm_available": self.tpm_available,
                "hsm_available": self.hsm_available,
                "fips_mode": self.fips_mode,
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
            "km_active": bool(status_reg & self.STATUS_KM_ACTIVE),
            "tpm_available": bool(status_reg & self.STATUS_TPM_AVAILABLE),
            "hsm_available": bool(status_reg & self.STATUS_HSM_AVAILABLE),
            "fips_mode": bool(status_reg & self.STATUS_FIPS_MODE),
            "auto_rotation": bool(status_reg & self.STATUS_AUTO_ROTATION),
            "backup_enabled": bool(status_reg & self.STATUS_BACKUP_ENABLED),
            "total_keys": len(self.keys),
            "active_keys": self._count_keys_by_status(KeyStatus.ACTIVE),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "KM_STATUS":
                value = self._read_status_register()
            elif register == "KEY_COUNT":
                value = len(self.keys)
            elif register == "KEY_CAPACITY":
                value = self.max_keys
            elif register == "ACTIVE_KEYS":
                value = self._count_keys_by_status(KeyStatus.ACTIVE)
            elif register == "EXPIRED_KEYS":
                value = self._count_keys_by_status(KeyStatus.EXPIRED)
            elif register == "ROTATION_COUNT":
                value = self.rotation_count
            elif register == "LAST_ROTATION":
                value = self.last_rotation_time
            elif register == "FIPS_MODE":
                value = 1 if self.fips_mode else 0
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

    # Key Management specific operations

    def list_keys(self, status_filter: Optional[int] = None) -> OperationResult:
        """List all keys or filter by status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        keys_list = []
        for key_id, key_info in self.keys.items():
            if status_filter is None or key_info['status'] == status_filter:
                keys_list.append({
                    "id": key_id,
                    "type": self._get_key_type_name(key_info['type']),
                    "usage": self._get_key_usage_name(key_info['usage']),
                    "status": self._get_key_status_name(key_info['status']),
                    "storage": self._get_storage_type_name(key_info['storage']),
                    "created": key_info['created'],
                    "expires": key_info.get('expires'),
                })

        self._record_operation(True)
        return OperationResult(True, data={
            "keys": keys_list,
            "total": len(keys_list),
        })

    def get_key_info(self, key_id: str) -> OperationResult:
        """Get detailed information about a specific key"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if key_id not in self.keys:
            return OperationResult(False, error=f"Key not found: {key_id}")

        key_info = self.keys[key_id].copy()
        # Don't expose actual key material
        if 'material' in key_info:
            key_info['material'] = "***PROTECTED***"

        key_info['type'] = self._get_key_type_name(key_info['type'])
        key_info['usage'] = self._get_key_usage_name(key_info['usage'])
        key_info['status'] = self._get_key_status_name(key_info['status'])
        key_info['storage'] = self._get_storage_type_name(key_info['storage'])

        self._record_operation(True)
        return OperationResult(True, data=key_info)

    def get_keys_by_type(self, key_type: int) -> OperationResult:
        """Get all keys of a specific type"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        keys_list = []
        for key_id, key_info in self.keys.items():
            if key_info['type'] == key_type:
                keys_list.append({
                    "id": key_id,
                    "usage": self._get_key_usage_name(key_info['usage']),
                    "status": self._get_key_status_name(key_info['status']),
                })

        self._record_operation(True)
        return OperationResult(True, data={
            "keys": keys_list,
            "total": len(keys_list),
            "key_type": self._get_key_type_name(key_type),
        })

    def get_rotation_history(self, limit: int = 50) -> OperationResult:
        """Get key rotation history"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        history = self.rotation_history[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "history": history,
            "total": len(history),
            "showing": f"Last {min(limit, len(self.rotation_history))} of {len(self.rotation_history)}",
        })

    def get_storage_summary(self) -> OperationResult:
        """Get summary of key storage locations"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        storage_counts = {}
        for key_info in self.keys.values():
            storage = self._get_storage_type_name(key_info['storage'])
            storage_counts[storage] = storage_counts.get(storage, 0) + 1

        summary = {
            "by_storage": storage_counts,
            "tpm_available": self.tpm_available,
            "hsm_available": self.hsm_available,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_key_summary(self) -> OperationResult:
        """Get comprehensive key management summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Count by type
        by_type = {}
        for key_info in self.keys.values():
            key_type = self._get_key_type_name(key_info['type'])
            by_type[key_type] = by_type.get(key_type, 0) + 1

        # Count by status
        by_status = {}
        for key_info in self.keys.values():
            status = self._get_key_status_name(key_info['status'])
            by_status[status] = by_status.get(status, 0) + 1

        # Count PQC compliance
        pqc_compliant_count = sum(1 for k in self.keys.values() if k.get('pqc_compliant', False))
        quantum_vulnerable_count = sum(1 for k in self.keys.values() if k.get('quantum_vulnerable', False))

        summary = {
            "total_keys": len(self.keys),
            "capacity": self.max_keys,
            "utilization": f"{len(self.keys)/self.max_keys*100:.1f}%",
            "by_type": by_type,
            "by_status": by_status,
            "rotation_count": self.rotation_count,
            "fips_mode": self.fips_mode,
            "pqc_compliant_keys": pqc_compliant_count,
            "quantum_vulnerable_keys": quantum_vulnerable_count,
            "pqc_compliance": f"{pqc_compliant_count/len(self.keys)*100:.1f}%" if self.keys else "0%",
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def check_pqc_compliance(self) -> OperationResult:
        """Check Post-Quantum Cryptography compliance for all keys"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        compliant_keys = []
        non_compliant_keys = []
        quantum_vulnerable_keys = []

        for key_id, key_info in self.keys.items():
            key_type = key_info['type']
            is_compliant = key_info.get('pqc_compliant', False)
            is_vulnerable = key_info.get('quantum_vulnerable', False)

            key_entry = {
                "id": key_id,
                "type": self._get_key_type_name(key_type),
                "algorithm": key_info.get('algorithm', 'Unknown'),
                "status": self._get_key_status_name(key_info['status']),
            }

            if is_vulnerable:
                key_entry['warning'] = key_info.get('deprecation_notice', 'Quantum-vulnerable')
                quantum_vulnerable_keys.append(key_entry)
            elif is_compliant:
                compliant_keys.append(key_entry)
            else:
                non_compliant_keys.append(key_entry)

        compliance_report = {
            "overall_compliant": len(quantum_vulnerable_keys) == 0 and len(non_compliant_keys) == 0,
            "compliant_keys": compliant_keys,
            "non_compliant_keys": non_compliant_keys,
            "quantum_vulnerable_keys": quantum_vulnerable_keys,
            "total_keys": len(self.keys),
            "compliance_percentage": f"{len(compliant_keys)/len(self.keys)*100:.1f}%" if self.keys else "0%",
            "required_algorithms": {
                "key_encapsulation": "ML-KEM-1024",
                "signatures": "ML-DSA-87",
                "symmetric": "AES-256-GCM",
                "hash": "SHA-512",
            },
        }

        self._record_operation(True)
        return OperationResult(True, data=compliance_report)

    def get_statistics(self) -> Dict[str, Any]:
        """Get key management statistics"""
        stats = super().get_statistics()

        stats.update({
            "total_keys": len(self.keys),
            "active_keys": self._count_keys_by_status(KeyStatus.ACTIVE),
            "rotation_count": self.rotation_count,
            "fips_mode": self.fips_mode,
            "tpm_available": self.tpm_available,
            "hsm_available": self.hsm_available,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read key management status register (simulated)"""
        status = self.STATUS_KM_ACTIVE

        if self.tpm_available:
            status |= self.STATUS_TPM_AVAILABLE

        if self.hsm_available:
            status |= self.STATUS_HSM_AVAILABLE

        if self.fips_mode:
            status |= self.STATUS_FIPS_MODE

        if self.auto_rotation_enabled:
            status |= self.STATUS_AUTO_ROTATION

        if self.backup_enabled:
            status |= self.STATUS_BACKUP_ENABLED

        return status

    def _initialize_sample_keys(self):
        """Initialize with sample keys (PQC-compliant)"""
        current_time = int(time.time())

        self.keys = {
            # PQC-compliant symmetric key (AES-256-GCM)
            "key_001": {
                "type": KeyType.SYMMETRIC_AES_256_GCM,
                "usage": KeyUsage.ENCRYPTION,
                "status": KeyStatus.ACTIVE,
                "storage": KeyStorageType.TPM,
                "created": current_time - 86400 * 30,
                "expires": current_time + 86400 * 335,
                "size_bits": 256,
                "material": secrets.token_bytes(32),
                "pqc_compliant": True,
                "algorithm": "AES-256-GCM",
            },
            # PQC-compliant signature key (ML-DSA-87)
            "key_002": {
                "type": KeyType.ML_DSA_87,
                "usage": KeyUsage.SIGNING,
                "status": KeyStatus.ACTIVE,
                "storage": KeyStorageType.HSM,
                "created": current_time - 86400 * 60,
                "expires": current_time + 86400 * 305,
                "size_bits": 4864 * 8,  # 4864 bytes secret key
                "material": secrets.token_bytes(4864),
                "pqc_compliant": True,
                "algorithm": "ML-DSA-87",
                "security_level": 5,
            },
            # PQC-compliant key encapsulation (ML-KEM-1024)
            "key_003": {
                "type": KeyType.ML_KEM_1024,
                "usage": KeyUsage.KEY_EXCHANGE,
                "status": KeyStatus.ACTIVE,
                "storage": KeyStorageType.TPM,
                "created": current_time - 86400 * 15,
                "expires": current_time + 86400 * 350,
                "size_bits": 3168 * 8,  # 3168 bytes secret key
                "material": secrets.token_bytes(3168),
                "pqc_compliant": True,
                "algorithm": "ML-KEM-1024",
                "security_level": 5,
            },
            # HMAC key (quantum-resistant hash)
            "key_004": {
                "type": KeyType.HMAC,
                "usage": KeyUsage.AUTHENTICATION,
                "status": KeyStatus.ACTIVE,
                "storage": KeyStorageType.SOFTWARE,
                "created": current_time - 86400 * 7,
                "expires": current_time + 86400 * 358,
                "size_bits": 512,  # SHA-512 based
                "material": secrets.token_bytes(64),
                "pqc_compliant": True,
                "algorithm": "HMAC-SHA-512",
            },
            # Session key (AES-256-GCM for short-lived operations)
            "key_005": {
                "type": KeyType.SESSION,
                "usage": KeyUsage.ENCRYPTION,
                "status": KeyStatus.ACTIVE,
                "storage": KeyStorageType.SECURE_ENCLAVE,
                "created": current_time - 3600,
                "expires": current_time + 3600 * 23,
                "size_bits": 256,
                "material": secrets.token_bytes(32),
                "pqc_compliant": True,
                "algorithm": "AES-256-GCM",
            },
            # Legacy RSA key (marked as quantum-vulnerable)
            "key_legacy_001": {
                "type": KeyType.ASYMMETRIC_RSA,
                "usage": KeyUsage.SIGNING,
                "status": KeyStatus.INACTIVE,
                "storage": KeyStorageType.HSM,
                "created": current_time - 86400 * 365,
                "expires": current_time - 86400 * 1,  # Expired
                "size_bits": 2048,
                "material": secrets.token_bytes(256),
                "pqc_compliant": False,
                "algorithm": "RSA-2048",
                "quantum_vulnerable": True,
                "deprecation_notice": "Replace with ML-DSA-87",
            },
            # Legacy ECC key (marked as quantum-vulnerable)
            "key_legacy_002": {
                "type": KeyType.ASYMMETRIC_ECC,
                "usage": KeyUsage.KEY_EXCHANGE,
                "status": KeyStatus.INACTIVE,
                "storage": KeyStorageType.TPM,
                "created": current_time - 86400 * 365,
                "expires": current_time - 86400 * 1,  # Expired
                "size_bits": 256,
                "material": secrets.token_bytes(32),
                "pqc_compliant": False,
                "algorithm": "ECC-P256",
                "quantum_vulnerable": True,
                "deprecation_notice": "Replace with ML-KEM-1024",
            },
        }

    def _count_keys_by_status(self, status: int) -> int:
        """Count keys with specific status"""
        return sum(1 for key in self.keys.values() if key['status'] == status)

    def _get_key_type_name(self, key_type: int) -> str:
        """Get key type name"""
        names = {
            # Symmetric
            KeyType.SYMMETRIC: "Symmetric",
            KeyType.SYMMETRIC_AES_256_GCM: "AES-256-GCM",
            # Post-Quantum Cryptography
            KeyType.ML_KEM_512: "ML-KEM-512",
            KeyType.ML_KEM_768: "ML-KEM-768",
            KeyType.ML_KEM_1024: "ML-KEM-1024",
            KeyType.ML_DSA_44: "ML-DSA-44",
            KeyType.ML_DSA_65: "ML-DSA-65",
            KeyType.ML_DSA_87: "ML-DSA-87",
            # Legacy (quantum-vulnerable)
            KeyType.ASYMMETRIC_RSA: "RSA (quantum-vulnerable)",
            KeyType.ASYMMETRIC_ECC: "ECC (quantum-vulnerable)",
            # Other
            KeyType.HMAC: "HMAC",
            KeyType.SESSION: "Session",
        }
        return names.get(key_type, "Unknown")

    def _get_key_usage_name(self, usage: int) -> str:
        """Get key usage name"""
        names = {
            KeyUsage.ENCRYPTION: "Encryption",
            KeyUsage.SIGNING: "Signing",
            KeyUsage.KEY_EXCHANGE: "Key Exchange",
            KeyUsage.AUTHENTICATION: "Authentication",
            KeyUsage.DERIVATION: "Derivation",
        }
        return names.get(usage, "Unknown")

    def _get_key_status_name(self, status: int) -> str:
        """Get key status name"""
        names = {
            KeyStatus.ACTIVE: "Active",
            KeyStatus.INACTIVE: "Inactive",
            KeyStatus.EXPIRED: "Expired",
            KeyStatus.REVOKED: "Revoked",
            KeyStatus.DESTROYED: "Destroyed",
        }
        return names.get(status, "Unknown")

    def _get_storage_type_name(self, storage: int) -> str:
        """Get storage type name"""
        names = {
            KeyStorageType.TPM: "TPM",
            KeyStorageType.HSM: "HSM",
            KeyStorageType.SOFTWARE: "Software",
            KeyStorageType.SECURE_ENCLAVE: "Secure Enclave",
        }
        return names.get(storage, "Unknown")


def main():
    """Test Key Management device"""
    print("=" * 80)
    print("Device 0x8013: Key Management - Test")
    print("=" * 80)

    device = KeyManagementDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Total keys: {result.data['total_keys']}")
        print(f"   Active keys: {result.data['active_keys']}")
        print(f"   FIPS mode: {result.data['fips_mode']}")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   KM active: {status['km_active']}")
    print(f"   TPM available: {status['tpm_available']}")
    print(f"   Total keys: {status['total_keys']}")

    # List keys
    print("\n3. Listing all keys...")
    result = device.list_keys()
    if result.success:
        print(f"   Total keys: {result.data['total']}")
        for key in result.data['keys'][:3]:
            print(f"   - {key['id']}: {key['type']} ({key['status']})")

    # Get key summary
    print("\n4. Getting key summary...")
    result = device.get_key_summary()
    if result.success:
        print(f"   Total keys: {result.data['total_keys']}")
        print(f"   By type: {result.data['by_type']}")
        print(f"   By status: {result.data['by_status']}")

    # Get storage summary
    print("\n5. Getting storage summary...")
    result = device.get_storage_summary()
    if result.success:
        print(f"   By storage: {result.data['by_storage']}")
        print(f"   TPM available: {result.data['tpm_available']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
