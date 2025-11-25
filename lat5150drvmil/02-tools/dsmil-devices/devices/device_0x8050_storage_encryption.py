#!/usr/bin/env python3
"""
Device 0x8050: Storage Encryption Controller

Full disk encryption, self-encrypting drive management, and secure storage
operations for military-grade data protection at rest.

Device ID: 0x8050
Group: 4 (Storage/Data)
Risk Level: MONITORED (Encryption configuration changes logged)

POST-QUANTUM CRYPTOGRAPHY COMPLIANCE:
- Symmetric Encryption: AES-256-GCM ONLY (FIPS 197 + SP 800-38D)
- Key Encapsulation: ML-KEM-1024 (FIPS 203)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)

⚠ Legacy XTS/CBC modes marked as deprecated
✓ All storage encryption uses AES-256-GCM with ML-KEM-1024 key encapsulation

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 2.0.0 (Post-Quantum)
"""

import sys
import os
import time

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from pqc_constants import (
    MLKEMAlgorithm,
    AESAlgorithm,
    HashAlgorithm,
    PQCProfile,
    REQUIRED_KEM,
    REQUIRED_SYMMETRIC,
    REQUIRED_HASH,
    is_pqc_compliant,
    get_algorithm_name,
)
from typing import Dict, List, Optional, Any


class EncryptionAlgorithm:
    """Storage encryption algorithms"""
    # PQC-compliant (REQUIRED for MIL-SPEC)
    AES_256_GCM = 0  # ⭐ REQUIRED - AES-256-GCM with ML-KEM-1024 key encapsulation

    # Legacy modes (DEPRECATED - not PQC-compliant)
    AES_256_XTS = 100  # ⚠ Deprecated - XTS mode not recommended for PQC
    AES_128_XTS = 101  # ⚠ Deprecated - insufficient key size and XTS mode
    CHACHA20_POLY1305 = 102  # ⚠ Deprecated - use AES-256-GCM


class VolumeStatus:
    """Volume encryption status"""
    UNENCRYPTED = 0
    ENCRYPTING = 1
    ENCRYPTED = 2
    DECRYPTING = 3
    LOCKED = 4
    UNLOCKED = 5


class SEDStatus:
    """Self-Encrypting Drive status"""
    NOT_SUPPORTED = 0
    DISABLED = 1
    ENABLED = 2
    LOCKED = 3
    UNLOCKED = 4


class StorageEncryptionDevice(DSMILDeviceBase):
    """Storage Encryption Controller (0x8050)"""

    # Register map
    REG_ENCRYPTION_STATUS = 0x00
    REG_ACTIVE_VOLUMES = 0x04
    REG_ENCRYPTED_VOLUMES = 0x08
    REG_SED_STATUS = 0x0C
    REG_ALGORITHM = 0x10
    REG_KEY_STRENGTH = 0x14
    REG_PERFORMANCE = 0x18
    REG_OPAL_VERSION = 0x1C

    # Status bits
    STATUS_ENCRYPTION_ACTIVE = 0x01
    STATUS_FDE_ENABLED = 0x02
    STATUS_SED_AVAILABLE = 0x04
    STATUS_OPAL_SUPPORTED = 0x08
    STATUS_FIPS_MODE = 0x10
    STATUS_SECURE_ERASE = 0x20
    STATUS_HW_CRYPTO = 0x40
    STATUS_KEY_PROTECTED = 0x80

    def __init__(self, device_id: int = 0x8050,
                 name: str = "Storage Encryption Controller",
                 description: str = "Full Disk Encryption and SED Management"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.volumes = {}
        self.sed_drives = {}
        self.fde_enabled = True
        self.fips_mode = True
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM  # PQC-compliant

        # Initialize default volume
        self._initialize_default_volumes()

        # Register map
        self.register_map = {
            "ENCRYPTION_STATUS": {
                "offset": self.REG_ENCRYPTION_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Encryption controller status"
            },
            "ACTIVE_VOLUMES": {
                "offset": self.REG_ACTIVE_VOLUMES,
                "size": 4,
                "access": "RO",
                "description": "Number of active volumes"
            },
            "ENCRYPTED_VOLUMES": {
                "offset": self.REG_ENCRYPTED_VOLUMES,
                "size": 4,
                "access": "RO",
                "description": "Number of encrypted volumes"
            },
            "SED_STATUS": {
                "offset": self.REG_SED_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Self-Encrypting Drive status"
            },
            "ALGORITHM": {
                "offset": self.REG_ALGORITHM,
                "size": 4,
                "access": "RW",
                "description": "Active encryption algorithm"
            },
            "KEY_STRENGTH": {
                "offset": self.REG_KEY_STRENGTH,
                "size": 4,
                "access": "RO",
                "description": "Encryption key strength (bits)"
            },
            "PERFORMANCE": {
                "offset": self.REG_PERFORMANCE,
                "size": 4,
                "access": "RO",
                "description": "Encryption performance (MB/s)"
            },
            "OPAL_VERSION": {
                "offset": self.REG_OPAL_VERSION,
                "size": 4,
                "access": "RO",
                "description": "OPAL specification version"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Storage Encryption Controller"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize with secure PQC-compliant defaults
            self.fde_enabled = True
            self.fips_mode = True
            self.default_algorithm = EncryptionAlgorithm.AES_256_GCM  # PQC-compliant
            self._initialize_default_volumes()
            self._initialize_sed_drives()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "fde_enabled": self.fde_enabled,
                "fips_mode": self.fips_mode,
                "algorithm": self._get_algorithm_name(self.default_algorithm),
                "volumes": len(self.volumes),
                "sed_drives": len(self.sed_drives),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_WRITE,
            DeviceCapability.CONFIGURATION,
            DeviceCapability.STATUS_REPORTING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        encrypted_count = sum(1 for v in self.volumes.values()
                            if v["status"] == VolumeStatus.ENCRYPTED)

        return {
            "encryption_active": bool(status_reg & self.STATUS_ENCRYPTION_ACTIVE),
            "fde_enabled": bool(status_reg & self.STATUS_FDE_ENABLED),
            "sed_available": bool(status_reg & self.STATUS_SED_AVAILABLE),
            "opal_supported": bool(status_reg & self.STATUS_OPAL_SUPPORTED),
            "fips_mode": bool(status_reg & self.STATUS_FIPS_MODE),
            "secure_erase": bool(status_reg & self.STATUS_SECURE_ERASE),
            "hw_crypto": bool(status_reg & self.STATUS_HW_CRYPTO),
            "key_protected": bool(status_reg & self.STATUS_KEY_PROTECTED),
            "total_volumes": len(self.volumes),
            "encrypted_volumes": encrypted_count,
            "sed_drives": len(self.sed_drives),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "ENCRYPTION_STATUS":
                value = self._read_status_register()
            elif register == "ACTIVE_VOLUMES":
                value = len(self.volumes)
            elif register == "ENCRYPTED_VOLUMES":
                value = sum(1 for v in self.volumes.values()
                          if v["status"] == VolumeStatus.ENCRYPTED)
            elif register == "SED_STATUS":
                value = SEDStatus.ENABLED if self.sed_drives else SEDStatus.NOT_SUPPORTED
            elif register == "ALGORITHM":
                value = self.default_algorithm
            elif register == "KEY_STRENGTH":
                # All PQC-compliant algorithms use 256-bit keys
                value = 256
            elif register == "PERFORMANCE":
                # Simulated encryption throughput
                value = 450  # MB/s
            elif register == "OPAL_VERSION":
                value = 0x0200  # OPAL 2.0
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

    # Storage Encryption specific operations

    def list_volumes(self) -> OperationResult:
        """List all storage volumes"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        volumes = []
        for volume_id, volume_info in self.volumes.items():
            volumes.append({
                "id": volume_id,
                "name": volume_info["name"],
                "path": volume_info["path"],
                "size_gb": volume_info["size_gb"],
                "status": self._get_volume_status_name(volume_info["status"]),
                "algorithm": self._get_algorithm_name(volume_info["algorithm"]),
                "encrypted": volume_info["status"] == VolumeStatus.ENCRYPTED,
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "volumes": volumes,
            "total": len(volumes),
            "encrypted": sum(1 for v in volumes if v["encrypted"]),
        })

    def get_volume_info(self, volume_id: str) -> OperationResult:
        """Get detailed volume information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if volume_id not in self.volumes:
            return OperationResult(False, error=f"Volume {volume_id} not found")

        volume = self.volumes[volume_id]

        self._record_operation(True)
        return OperationResult(True, data={
            "volume_id": volume_id,
            "name": volume["name"],
            "path": volume["path"],
            "size_gb": volume["size_gb"],
            "status": self._get_volume_status_name(volume["status"]),
            "algorithm": self._get_algorithm_name(volume["algorithm"]),
            "key_strength": volume["key_strength"],
            "encrypted": volume["status"] == VolumeStatus.ENCRYPTED,
            "locked": volume["status"] == VolumeStatus.LOCKED,
            "progress": volume.get("progress", 100),
        })

    def list_sed_drives(self) -> OperationResult:
        """List Self-Encrypting Drives"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        drives = []
        for drive_id, drive_info in self.sed_drives.items():
            drives.append({
                "id": drive_id,
                "model": drive_info["model"],
                "serial": drive_info["serial"],
                "capacity_gb": drive_info["capacity_gb"],
                "status": self._get_sed_status_name(drive_info["status"]),
                "opal_version": drive_info.get("opal_version", "2.0"),
                "locked": drive_info["status"] == SEDStatus.LOCKED,
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "drives": drives,
            "total": len(drives),
        })

    def get_sed_info(self, drive_id: str) -> OperationResult:
        """Get detailed SED information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if drive_id not in self.sed_drives:
            return OperationResult(False, error=f"SED drive {drive_id} not found")

        drive = self.sed_drives[drive_id]

        self._record_operation(True)
        return OperationResult(True, data={
            "drive_id": drive_id,
            "model": drive["model"],
            "serial": drive["serial"],
            "capacity_gb": drive["capacity_gb"],
            "status": self._get_sed_status_name(drive["status"]),
            "opal_version": drive.get("opal_version", "2.0"),
            "firmware": drive.get("firmware", "Unknown"),
            "hardware_crypto": True,
            "locked": drive["status"] == SEDStatus.LOCKED,
        })

    def get_encryption_config(self) -> OperationResult:
        """Get encryption configuration"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        config = {
            "fde_enabled": self.fde_enabled,
            "fips_mode": self.fips_mode,
            "pqc_compliant": True,
            "default_algorithm": self._get_algorithm_name(self.default_algorithm),
            "key_strength": 256,
            "key_encapsulation": "ML-KEM-1024",
            "hash_function": "SHA-512",
            "hardware_crypto": True,
            "algorithms_available": [
                "AES-256-GCM (PQC-compliant)",
            ],
            "legacy_algorithms_deprecated": [
                "AES-256-XTS (not recommended)",
                "AES-128-XTS (not recommended)",
                "ChaCha20-Poly1305 (not recommended)",
            ],
        }

        self._record_operation(True)
        return OperationResult(True, data=config)

    def get_encryption_performance(self) -> OperationResult:
        """Get encryption performance metrics"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Simulated performance metrics
        performance = {
            "read_throughput_mbps": 520,
            "write_throughput_mbps": 450,
            "cpu_overhead_percent": 5,
            "hardware_accelerated": True,
            "aes_ni_available": True,
        }

        self._record_operation(True)
        return OperationResult(True, data=performance)

    def get_key_management_info(self) -> OperationResult:
        """Get key management information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        key_info = {
            "key_storage": "TPM 2.0",
            "key_derivation": "PBKDF2-HMAC-SHA512",
            "key_encapsulation": "ML-KEM-1024",
            "pqc_compliant": True,
            "key_escrow": False,
            "recovery_key_available": True,
            "key_rotation_supported": True,
            "last_rotation": "2025-01-01T00:00:00Z",
        }

        self._record_operation(True)
        return OperationResult(True, data=key_info)

    def check_pqc_compliance(self) -> OperationResult:
        """Check Post-Quantum Cryptography compliance for all volumes"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        compliant_volumes = []
        non_compliant_volumes = []

        for volume_id, volume_info in self.volumes.items():
            is_compliant = volume_info.get('pqc_compliant', False)

            volume_entry = {
                "id": volume_id,
                "name": volume_info["name"],
                "algorithm": self._get_algorithm_name(volume_info["algorithm"]),
                "status": self._get_volume_status_name(volume_info["status"]),
                "key_strength": volume_info["key_strength"],
            }

            if is_compliant:
                volume_entry["key_encapsulation"] = volume_info.get("key_encapsulation", "N/A")
                volume_entry["hash_function"] = volume_info.get("hash_function", "N/A")
                compliant_volumes.append(volume_entry)
            else:
                volume_entry["warning"] = volume_info.get("deprecation_notice", "Not PQC-compliant")
                non_compliant_volumes.append(volume_entry)

        compliance_report = {
            "overall_compliant": len(non_compliant_volumes) == 0,
            "compliant_volumes": compliant_volumes,
            "non_compliant_volumes": non_compliant_volumes,
            "total_volumes": len(self.volumes),
            "compliance_percentage": f"{len(compliant_volumes)/len(self.volumes)*100:.1f}%" if self.volumes else "0%",
            "required_algorithm": "AES-256-GCM",
            "required_key_encapsulation": "ML-KEM-1024",
            "required_hash": "SHA-512",
        }

        self._record_operation(True)
        return OperationResult(True, data=compliance_report)

    def get_opal_support(self) -> OperationResult:
        """Get OPAL specification support"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        opal_info = {
            "supported": len(self.sed_drives) > 0,
            "version": "2.0",
            "features": [
                "Locking",
                "MBR Shadowing",
                "Single User Mode",
                "DataStore Tables",
                "Admin SP",
                "Locking SP",
            ] if len(self.sed_drives) > 0 else [],
        }

        self._record_operation(True)
        return OperationResult(True, data=opal_info)

    def get_statistics(self) -> Dict[str, Any]:
        """Get encryption statistics"""
        stats = super().get_statistics()

        encrypted_count = sum(1 for v in self.volumes.values()
                            if v["status"] == VolumeStatus.ENCRYPTED)

        stats.update({
            "total_volumes": len(self.volumes),
            "encrypted_volumes": encrypted_count,
            "encryption_percentage": (encrypted_count / len(self.volumes) * 100) if self.volumes else 0,
            "sed_drives": len(self.sed_drives),
            "fde_enabled": self.fde_enabled,
            "fips_mode": self.fips_mode,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read encryption status register (simulated)"""
        status = 0

        if any(v["status"] == VolumeStatus.ENCRYPTED for v in self.volumes.values()):
            status |= self.STATUS_ENCRYPTION_ACTIVE

        if self.fde_enabled:
            status |= self.STATUS_FDE_ENABLED

        if self.sed_drives:
            status |= self.STATUS_SED_AVAILABLE
            status |= self.STATUS_OPAL_SUPPORTED

        if self.fips_mode:
            status |= self.STATUS_FIPS_MODE

        status |= self.STATUS_SECURE_ERASE
        status |= self.STATUS_HW_CRYPTO
        status |= self.STATUS_KEY_PROTECTED

        return status

    def _initialize_default_volumes(self):
        """Initialize default volumes (PQC-compliant)"""
        self.volumes = {
            "volume0": {
                "name": "System Volume",
                "path": "/dev/nvme0n1p1",
                "size_gb": 256,
                "status": VolumeStatus.ENCRYPTED,
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_strength": 256,
                "progress": 100,
                "pqc_compliant": True,
                "key_encapsulation": "ML-KEM-1024",
                "hash_function": "SHA-512",
            },
            "volume1": {
                "name": "Data Volume",
                "path": "/dev/nvme0n1p2",
                "size_gb": 512,
                "status": VolumeStatus.ENCRYPTED,
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_strength": 256,
                "progress": 100,
                "pqc_compliant": True,
                "key_encapsulation": "ML-KEM-1024",
                "hash_function": "SHA-512",
            },
            # Legacy volume (for backward compatibility testing)
            "volume_legacy": {
                "name": "Legacy Volume (deprecated)",
                "path": "/dev/sda1",
                "size_gb": 128,
                "status": VolumeStatus.LOCKED,
                "algorithm": EncryptionAlgorithm.AES_256_XTS,
                "key_strength": 256,
                "progress": 100,
                "pqc_compliant": False,
                "deprecation_notice": "Migrate to AES-256-GCM with ML-KEM-1024",
            },
        }

    def _initialize_sed_drives(self):
        """Initialize SED drives"""
        self.sed_drives = {
            "sed0": {
                "model": "Samsung 980 PRO",
                "serial": "S5GXNX0R123456",
                "capacity_gb": 1024,
                "status": SEDStatus.UNLOCKED,
                "opal_version": "2.0",
                "firmware": "5B2QGXA7",
            }
        }

    def _get_algorithm_name(self, algorithm: int) -> str:
        """Get algorithm name"""
        names = {
            # PQC-compliant
            EncryptionAlgorithm.AES_256_GCM: "AES-256-GCM",
            # Legacy (deprecated)
            EncryptionAlgorithm.AES_256_XTS: "AES-256-XTS (deprecated)",
            EncryptionAlgorithm.AES_128_XTS: "AES-128-XTS (deprecated)",
            EncryptionAlgorithm.CHACHA20_POLY1305: "ChaCha20-Poly1305 (deprecated)",
        }
        return names.get(algorithm, "Unknown")

    def _get_volume_status_name(self, status: int) -> str:
        """Get volume status name"""
        names = {
            VolumeStatus.UNENCRYPTED: "Unencrypted",
            VolumeStatus.ENCRYPTING: "Encrypting",
            VolumeStatus.ENCRYPTED: "Encrypted",
            VolumeStatus.DECRYPTING: "Decrypting",
            VolumeStatus.LOCKED: "Locked",
            VolumeStatus.UNLOCKED: "Unlocked",
        }
        return names.get(status, "Unknown")

    def _get_sed_status_name(self, status: int) -> str:
        """Get SED status name"""
        names = {
            SEDStatus.NOT_SUPPORTED: "Not Supported",
            SEDStatus.DISABLED: "Disabled",
            SEDStatus.ENABLED: "Enabled",
            SEDStatus.LOCKED: "Locked",
            SEDStatus.UNLOCKED: "Unlocked",
        }
        return names.get(status, "Unknown")
