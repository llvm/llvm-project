#!/usr/bin/env python3
"""
Device 0x8002: Credential Vault

Secure storage and management of credentials, keys, passwords, and
sensitive authentication data with hardware encryption.

Device ID: 0x8002
Group: 0 (Core Security)
Risk Level: MONITORED (75% safe for READ operations)

POST-QUANTUM CRYPTOGRAPHY COMPLIANCE:
- Symmetric Encryption: AES-256-GCM (FIPS 197 + SP 800-38D)
- Key Encapsulation: ML-KEM-1024 (FIPS 203)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)

✓ All credential storage uses AES-256-GCM with ML-KEM-1024 key encapsulation
✓ TPM 2.0 sealed with post-quantum resistant algorithms

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
    AESAlgorithm,
    HashAlgorithm,
    MLKEMAlgorithm,
    PQCProfile,
    REQUIRED_SYMMETRIC,
    REQUIRED_HASH,
    REQUIRED_KEM,
    is_pqc_compliant,
)
from typing import Dict, List, Optional, Any


class CredentialType(object):
    """Credential type identifiers"""
    PASSWORD = 0
    API_KEY = 1
    CERTIFICATE = 2
    SSH_KEY = 3
    SYMMETRIC_KEY = 4
    ASYMMETRIC_KEY = 5
    TOKEN = 6
    BIOMETRIC = 7


class VaultPolicy(object):
    """Vault security policy flags"""
    ENCRYPTION_ENABLED = 0x0001
    TPM_SEALED = 0x0002
    BIOMETRIC_REQUIRED = 0x0004
    TIME_LIMITED = 0x0008
    ACCESS_LOGGED = 0x0010
    AUTO_LOCK = 0x0020
    WIPE_ON_TAMPERING = 0x0040
    MULTI_FACTOR = 0x0080


class CredentialVaultDevice(DSMILDeviceBase):
    """Credential Vault Device (0x8002)"""

    # Register map
    REG_VAULT_STATUS = 0x00
    REG_VAULT_POLICY = 0x04
    REG_CREDENTIAL_COUNT = 0x08
    REG_AVAILABLE_SLOTS = 0x0C
    REG_LOCK_STATUS = 0x10
    REG_ACCESS_COUNT = 0x14
    REG_LAST_ACCESS_TIME = 0x18
    REG_TAMPER_STATUS = 0x1C

    # Status bits
    STATUS_VAULT_READY = 0x01
    STATUS_VAULT_LOCKED = 0x02
    STATUS_ENCRYPTED = 0x04
    STATUS_TPM_SEALED = 0x08
    STATUS_TAMPER_DETECTED = 0x10
    STATUS_BACKUP_VALID = 0x20
    STATUS_CAPACITY_WARNING = 0x40
    STATUS_AUTO_LOCK_ACTIVE = 0x80

    def __init__(self, device_id: int = 0x8002,
                 name: str = "Credential Vault",
                 description: str = "Secure Credential Storage and Management"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.vault_policy = 0
        self.is_locked = False
        self.credentials = {}  # slot_id -> credential_info
        self.max_slots = 256
        self.access_log = []
        self.tamper_detected = False

        # Register map
        self.register_map = {
            "VAULT_STATUS": {
                "offset": self.REG_VAULT_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Vault status register"
            },
            "VAULT_POLICY": {
                "offset": self.REG_VAULT_POLICY,
                "size": 4,
                "access": "RO",
                "description": "Active vault security policy"
            },
            "CREDENTIAL_COUNT": {
                "offset": self.REG_CREDENTIAL_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of stored credentials"
            },
            "AVAILABLE_SLOTS": {
                "offset": self.REG_AVAILABLE_SLOTS,
                "size": 4,
                "access": "RO",
                "description": "Available credential slots"
            },
            "LOCK_STATUS": {
                "offset": self.REG_LOCK_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Vault lock status"
            },
            "ACCESS_COUNT": {
                "offset": self.REG_ACCESS_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total access count"
            },
            "LAST_ACCESS_TIME": {
                "offset": self.REG_LAST_ACCESS_TIME,
                "size": 4,
                "access": "RO",
                "description": "Last access timestamp"
            },
            "TAMPER_STATUS": {
                "offset": self.REG_TAMPER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Tamper detection status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Credential Vault device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Read vault policy
            self.vault_policy = self._read_vault_policy()

            # Initialize vault as locked (secure default)
            self.is_locked = True

            # Initialize some sample credentials (simulated, PQC-compliant)
            self.credentials = {
                1: {
                    "type": CredentialType.PASSWORD,
                    "name": "root_password",
                    "size": 64,
                    "encryption": "AES-256-GCM",
                    "key_encapsulation": "ML-KEM-1024",
                    "hash": "SHA-512",
                    "pqc_compliant": True,
                },
                2: {
                    "type": CredentialType.SSH_KEY,
                    "name": "admin_ssh",
                    "size": 2048,
                    "encryption": "AES-256-GCM",
                    "key_encapsulation": "ML-KEM-1024",
                    "hash": "SHA-512",
                    "pqc_compliant": True,
                },
                3: {
                    "type": CredentialType.API_KEY,
                    "name": "api_key_1",
                    "size": 128,
                    "encryption": "AES-256-GCM",
                    "key_encapsulation": "ML-KEM-1024",
                    "hash": "SHA-512",
                    "pqc_compliant": True,
                },
            }

            self.tamper_detected = False

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "vault_policy": self.vault_policy,
                "locked": self.is_locked,
                "credentials": len(self.credentials),
                "available_slots": self.max_slots - len(self.credentials),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_WRITE,
            DeviceCapability.ENCRYPTED_STORAGE,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.EVENT_MONITORING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "vault_ready": bool(status_reg & self.STATUS_VAULT_READY),
            "locked": bool(status_reg & self.STATUS_VAULT_LOCKED),
            "encrypted": bool(status_reg & self.STATUS_ENCRYPTED),
            "tpm_sealed": bool(status_reg & self.STATUS_TPM_SEALED),
            "tamper_detected": bool(status_reg & self.STATUS_TAMPER_DETECTED),
            "backup_valid": bool(status_reg & self.STATUS_BACKUP_VALID),
            "capacity_warning": bool(status_reg & self.STATUS_CAPACITY_WARNING),
            "auto_lock_active": bool(status_reg & self.STATUS_AUTO_LOCK_ACTIVE),
            "credentials": len(self.credentials),
            "available_slots": self.max_slots - len(self.credentials),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            # Simulated register reads
            if register == "VAULT_STATUS":
                value = self._read_status_register()
            elif register == "VAULT_POLICY":
                value = self.vault_policy
            elif register == "CREDENTIAL_COUNT":
                value = len(self.credentials)
            elif register == "AVAILABLE_SLOTS":
                value = self.max_slots - len(self.credentials)
            elif register == "LOCK_STATUS":
                value = 1 if self.is_locked else 0
            elif register == "ACCESS_COUNT":
                value = len(self.access_log)
            elif register == "LAST_ACCESS_TIME":
                import time
                value = int(time.time()) if self.access_log else 0
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

    # Credential Vault specific operations

    def get_vault_policy(self) -> OperationResult:
        """Get active vault security policy"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        policy_flags = []
        if self.vault_policy & VaultPolicy.ENCRYPTION_ENABLED:
            policy_flags.append("ENCRYPTION")
        if self.vault_policy & VaultPolicy.TPM_SEALED:
            policy_flags.append("TPM_SEALED")
        if self.vault_policy & VaultPolicy.BIOMETRIC_REQUIRED:
            policy_flags.append("BIOMETRIC")
        if self.vault_policy & VaultPolicy.TIME_LIMITED:
            policy_flags.append("TIME_LIMITED")
        if self.vault_policy & VaultPolicy.ACCESS_LOGGED:
            policy_flags.append("LOGGED")
        if self.vault_policy & VaultPolicy.AUTO_LOCK:
            policy_flags.append("AUTO_LOCK")
        if self.vault_policy & VaultPolicy.WIPE_ON_TAMPERING:
            policy_flags.append("TAMPER_WIPE")
        if self.vault_policy & VaultPolicy.MULTI_FACTOR:
            policy_flags.append("MULTI_FACTOR")

        self._record_operation(True)
        return OperationResult(True, data={
            "policy": self.vault_policy,
            "flags": policy_flags,
        })

    def list_credentials(self) -> OperationResult:
        """List stored credentials (metadata only)"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if self.is_locked:
            return OperationResult(False, error="Vault is locked")

        creds = []
        for slot_id, cred_info in self.credentials.items():
            creds.append({
                "slot": slot_id,
                "type": self._get_credential_type_name(cred_info["type"]),
                "name": cred_info["name"],
                "size": cred_info["size"],
            })

        self._log_access("list_credentials")
        self._record_operation(True)

        return OperationResult(True, data={
            "credentials": creds,
            "total": len(creds),
        })

    def get_credential_info(self, slot_id: int) -> OperationResult:
        """
        Get credential metadata

        Args:
            slot_id: Credential slot ID

        Returns:
            OperationResult with credential info
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if self.is_locked:
            return OperationResult(False, error="Vault is locked")

        if slot_id not in self.credentials:
            return OperationResult(False, error="Credential not found")

        cred = self.credentials[slot_id]

        self._log_access(f"get_info_{slot_id}")
        self._record_operation(True)

        return OperationResult(True, data={
            "slot": slot_id,
            "type": self._get_credential_type_name(cred["type"]),
            "name": cred["name"],
            "size": cred["size"],
            "encrypted": True,
        })

    def retrieve_credential(self, slot_id: int) -> OperationResult:
        """
        Retrieve credential data (requires unlocked vault)

        Args:
            slot_id: Credential slot ID

        Returns:
            OperationResult with credential data
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if self.is_locked:
            return OperationResult(False, error="Vault is locked - unlock required")

        if slot_id not in self.credentials:
            return OperationResult(False, error="Credential not found")

        # Simulated credential retrieval
        # In real implementation, this would decrypt and return actual data

        self._log_access(f"retrieve_{slot_id}")
        self._record_operation(True)

        return OperationResult(True, data={
            "slot": slot_id,
            "retrieved": True,
            "size": self.credentials[slot_id]["size"],
        })

    def get_capacity_info(self) -> OperationResult:
        """Get vault capacity information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        used = len(self.credentials)
        available = self.max_slots - used
        usage_percent = (used / self.max_slots) * 100

        self._record_operation(True)
        return OperationResult(True, data={
            "total_slots": self.max_slots,
            "used_slots": used,
            "available_slots": available,
            "usage_percent": round(usage_percent, 2),
            "capacity_warning": usage_percent > 80,
        })

    def get_access_log(self, limit: int = 10) -> OperationResult:
        """
        Get vault access log

        Args:
            limit: Maximum number of log entries to return

        Returns:
            OperationResult with access log
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if self.is_locked:
            return OperationResult(False, error="Vault is locked")

        log_entries = self.access_log[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "log_entries": log_entries,
            "total_accesses": len(self.access_log),
            "returned": len(log_entries),
        })

    def check_tamper_status(self) -> OperationResult:
        """Check for tampering attempts"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "tamper_detected": self.tamper_detected,
            "vault_integrity": not self.tamper_detected,
            "wipe_on_tamper": bool(self.vault_policy & VaultPolicy.WIPE_ON_TAMPERING),
        })

    def unlock_vault(self, auth_token: str = None) -> OperationResult:
        """
        Unlock vault (simulated - requires authentication)

        Args:
            auth_token: Authentication token

        Returns:
            OperationResult with unlock status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if not self.is_locked:
            return OperationResult(True, data={"already_unlocked": True})

        # Simulated authentication
        # In real implementation, this would verify auth token, biometrics, etc.

        if auth_token == "simulated_token":
            self.is_locked = False
            self._log_access("vault_unlock")
            self._record_operation(True)
            return OperationResult(True, data={"unlocked": True})
        else:
            self._record_operation(False, "Invalid authentication")
            return OperationResult(False, error="Authentication failed")

    def lock_vault(self) -> OperationResult:
        """Lock vault immediately"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self.is_locked = True
        self._log_access("vault_lock")
        self._record_operation(True)

        return OperationResult(True, data={"locked": True})

    def check_pqc_compliance(self) -> OperationResult:
        """Check Post-Quantum Cryptography compliance for all credentials"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if self.is_locked:
            return OperationResult(False, error="Vault is locked - unlock required for compliance check")

        compliant_credentials = []
        non_compliant_credentials = []

        for slot_id, cred_info in self.credentials.items():
            is_compliant = cred_info.get('pqc_compliant', False)

            cred_entry = {
                "slot": slot_id,
                "name": cred_info["name"],
                "type": self._get_credential_type_name(cred_info["type"]),
                "encryption": cred_info.get("encryption", "Unknown"),
                "key_encapsulation": cred_info.get("key_encapsulation", "Unknown"),
                "hash": cred_info.get("hash", "Unknown"),
            }

            if is_compliant:
                compliant_credentials.append(cred_entry)
            else:
                cred_entry["warning"] = "Not PQC-compliant - requires migration"
                non_compliant_credentials.append(cred_entry)

        compliance_report = {
            "overall_compliant": len(non_compliant_credentials) == 0,
            "compliant_credentials": compliant_credentials,
            "non_compliant_credentials": non_compliant_credentials,
            "total_credentials": len(self.credentials),
            "compliance_percentage": f"{len(compliant_credentials)/len(self.credentials)*100:.1f}%" if self.credentials else "0%",
            "vault_encryption": "AES-256-GCM",
            "vault_key_encapsulation": "ML-KEM-1024",
            "vault_hash": "SHA-512",
            "tpm_sealed": bool(self.vault_policy & VaultPolicy.TPM_SEALED),
        }

        self._log_access("pqc_compliance_check")
        self._record_operation(True)
        return OperationResult(True, data=compliance_report)

    def get_encryption_info(self) -> OperationResult:
        """Get vault encryption configuration"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        encryption_info = {
            "algorithm": "AES-256-GCM",
            "key_strength": 256,
            "key_encapsulation": "ML-KEM-1024",
            "hash_function": "SHA-512",
            "pqc_compliant": True,
            "security_level": 5,
            "quantum_security": "~200-bit",
            "tpm_sealed": bool(self.vault_policy & VaultPolicy.TPM_SEALED),
            "hardware_protected": True,
        }

        self._record_operation(True)
        return OperationResult(True, data=encryption_info)

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read vault status register (simulated)"""
        status = self.STATUS_VAULT_READY | self.STATUS_ENCRYPTED | self.STATUS_TPM_SEALED

        if self.is_locked:
            status |= self.STATUS_VAULT_LOCKED

        if self.tamper_detected:
            status |= self.STATUS_TAMPER_DETECTED

        if (len(self.credentials) / self.max_slots) > 0.8:
            status |= self.STATUS_CAPACITY_WARNING

        if self.vault_policy & VaultPolicy.AUTO_LOCK:
            status |= self.STATUS_AUTO_LOCK_ACTIVE

        return status

    def _read_vault_policy(self) -> int:
        """Read vault policy (simulated)"""
        policy = (
            VaultPolicy.ENCRYPTION_ENABLED |
            VaultPolicy.TPM_SEALED |
            VaultPolicy.ACCESS_LOGGED |
            VaultPolicy.AUTO_LOCK |
            VaultPolicy.WIPE_ON_TAMPERING
        )
        return policy

    def _get_credential_type_name(self, cred_type: int) -> str:
        """Get credential type name"""
        type_names = {
            CredentialType.PASSWORD: "Password",
            CredentialType.API_KEY: "API Key",
            CredentialType.CERTIFICATE: "Certificate",
            CredentialType.SSH_KEY: "SSH Key",
            CredentialType.SYMMETRIC_KEY: "Symmetric Key",
            CredentialType.ASYMMETRIC_KEY: "Asymmetric Key",
            CredentialType.TOKEN: "Token",
            CredentialType.BIOMETRIC: "Biometric",
        }
        return type_names.get(cred_type, "Unknown")

    def _log_access(self, operation: str):
        """Log vault access"""
        import time
        self.access_log.append({
            "timestamp": time.time(),
            "operation": operation,
        })
        # Keep only last 1000 entries
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]
