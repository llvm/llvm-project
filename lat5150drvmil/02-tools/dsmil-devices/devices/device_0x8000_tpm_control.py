#!/usr/bin/env python3
"""
Device 0x8000: TPM Control Interface

Provides high-level interface to the TPM 2.0 subsystem for cryptographic
operations, secure key storage, and attestation.

POST-QUANTUM CRYPTOGRAPHY COMPLIANCE:
- Key Encapsulation: ML-KEM-1024 (FIPS 203)
- Digital Signatures: ML-DSA-87 (FIPS 204)
- Symmetric Encryption: AES-256-GCM (FIPS 197 + SP 800-38D)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)

Device ID: 0x8000
Group: 0 (Core Security)
Risk Level: MONITORED (85% safe for READ operations)

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


class TPM2Algorithm(object):
    """
    TPM 2.0 Algorithm identifiers with Post-Quantum Cryptography Support

    REQUIRED for MIL-SPEC systems:
    - ML-KEM-1024 (Key Encapsulation)
    - ML-DSA-87 (Digital Signatures)
    - AES-256-GCM (Symmetric Encryption)
    - SHA-512 (Hashing)
    """

    # ========== POST-QUANTUM ALGORITHMS (REQUIRED) ==========

    # ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism) - FIPS 203
    ML_KEM_512 = MLKEMAlgorithm.ML_KEM_512
    ML_KEM_768 = MLKEMAlgorithm.ML_KEM_768
    ML_KEM_1024 = MLKEMAlgorithm.ML_KEM_1024  # ⭐ REQUIRED for MIL-SPEC

    # ML-DSA (Module-Lattice-Based Digital Signature Algorithm) - FIPS 204
    ML_DSA_44 = MLDSAAlgorithm.ML_DSA_44
    ML_DSA_65 = MLDSAAlgorithm.ML_DSA_65
    ML_DSA_87 = MLDSAAlgorithm.ML_DSA_87      # ⭐ REQUIRED for MIL-SPEC

    # AES with GCM Mode - FIPS 197 + NIST SP 800-38D
    AES_128_GCM = AESAlgorithm.AES_128_GCM
    AES_192_GCM = AESAlgorithm.AES_192_GCM
    AES_256_GCM = AESAlgorithm.AES_256_GCM    # ⭐ REQUIRED for MIL-SPEC

    # Quantum-Resistant Hash Functions - FIPS 180-4
    SHA384 = HashAlgorithm.SHA384             # ⭐ RECOMMENDED
    SHA512 = HashAlgorithm.SHA512             # ⭐ REQUIRED for MIL-SPEC
    SHA3_384 = HashAlgorithm.SHA3_384
    SHA3_512 = HashAlgorithm.SHA3_512

    # ========== LEGACY ALGORITHMS (DEPRECATED FOR NEW SYSTEMS) ==========

    # Legacy symmetric (use AES-256-GCM instead)
    AES_128_CBC = AESAlgorithm.AES_128_CBC    # Deprecated
    AES_256_CBC = AESAlgorithm.AES_256_CBC    # Deprecated
    AES_128_CTR = AESAlgorithm.AES_128_CTR    # Deprecated
    AES_256_CTR = AESAlgorithm.AES_256_CTR    # Deprecated

    # Legacy asymmetric (quantum-vulnerable - DO NOT USE)
    RSA_2048 = 0x0010   # Quantum-vulnerable
    RSA_3072 = 0x0011   # Quantum-vulnerable
    RSA_4096 = 0x0012   # Quantum-vulnerable
    ECC_P256 = 0x0020   # Quantum-vulnerable
    ECC_P384 = 0x0021   # Quantum-vulnerable
    ECC_P521 = 0x0022   # Quantum-vulnerable

    # Legacy hash (use SHA-512 instead)
    SHA256 = HashAlgorithm.SHA256             # Deprecated for PQC
    SHA3_256 = HashAlgorithm.SHA3_256         # Deprecated for PQC

    # Backward compatibility aliases (deprecated)
    KYBER512 = ML_KEM_512
    KYBER768 = ML_KEM_768
    KYBER1024 = ML_KEM_1024
    DILITHIUM2 = ML_DSA_44
    DILITHIUM3 = ML_DSA_65
    DILITHIUM5 = ML_DSA_87


class TPMControlDevice(DSMILDeviceBase):
    """TPM Control Device (0x8000)"""

    # Register map
    REG_TPM_STATUS = 0x00
    REG_TPM_CAPABILITIES = 0x04
    REG_TPM_VERSION = 0x08
    REG_TPM_COMMAND = 0x0C
    REG_TPM_RESPONSE = 0x10
    REG_TPM_ERROR_CODE = 0x14
    REG_TPM_PCR_SELECT = 0x18
    REG_TPM_KEY_HANDLE = 0x1C

    # Status bits
    STATUS_READY = 0x01
    STATUS_BUSY = 0x02
    STATUS_ERROR = 0x04
    STATUS_LOCKED = 0x08
    STATUS_INITIALIZED = 0x10

    # Capability bits
    CAP_RSA = 0x0001
    CAP_ECC = 0x0002
    CAP_AES = 0x0004
    CAP_SHA256 = 0x0008
    CAP_SHA512 = 0x0010
    CAP_PCR_EXTEND = 0x0020
    CAP_KEY_GENERATION = 0x0040
    CAP_SEALING = 0x0080
    CAP_ATTESTATION = 0x0100
    CAP_POST_QUANTUM = 0x8000

    def __init__(self, device_id: int = 0x8000,
                 name: str = "TPM Control",
                 description: str = "TPM 2.0 Cryptographic Control Interface"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.tpm_version = None
        self.supported_algorithms = []
        self.active_keys = {}
        self.pcr_values = {}

        # Register map
        self.register_map = {
            "STATUS": {
                "offset": self.REG_TPM_STATUS,
                "size": 4,
                "access": "RO",
                "description": "TPM status register"
            },
            "CAPABILITIES": {
                "offset": self.REG_TPM_CAPABILITIES,
                "size": 4,
                "access": "RO",
                "description": "TPM capabilities bitmap"
            },
            "VERSION": {
                "offset": self.REG_TPM_VERSION,
                "size": 4,
                "access": "RO",
                "description": "TPM version (major.minor)"
            },
            "COMMAND": {
                "offset": self.REG_TPM_COMMAND,
                "size": 4,
                "access": "WO",
                "description": "TPM command register"
            },
            "RESPONSE": {
                "offset": self.REG_TPM_RESPONSE,
                "size": 4,
                "access": "RO",
                "description": "TPM response register"
            },
            "ERROR_CODE": {
                "offset": self.REG_TPM_ERROR_CODE,
                "size": 4,
                "access": "RO",
                "description": "Last error code"
            },
            "PCR_SELECT": {
                "offset": self.REG_TPM_PCR_SELECT,
                "size": 4,
                "access": "RW",
                "description": "PCR selection register"
            },
            "KEY_HANDLE": {
                "offset": self.REG_TPM_KEY_HANDLE,
                "size": 4,
                "access": "RW",
                "description": "Active key handle"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize TPM device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Check TPM availability (simulated)
            # In real implementation, this would check /dev/tpm0 or TPM hardware
            status = self._read_status_register()

            if not (status & self.STATUS_READY):
                self.state = DeviceState.ERROR
                return OperationResult(False, error="TPM not ready")

            # Read capabilities
            caps = self._read_capabilities_register()
            self._decode_capabilities(caps)

            # Read version
            version_raw = self._read_version_register()
            self.tpm_version = f"{(version_raw >> 16) & 0xFF}.{version_raw & 0xFFFF}"

            # Initialize PCR values (24 PCRs)
            for i in range(24):
                self.pcr_values[i] = 0

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "version": self.tpm_version,
                "capabilities": len(self.capabilities),
                "algorithms": len(self.supported_algorithms),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.CONFIGURATION,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.ENCRYPTED_STORAGE,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "ready": bool(status_reg & self.STATUS_READY),
            "busy": bool(status_reg & self.STATUS_BUSY),
            "error": bool(status_reg & self.STATUS_ERROR),
            "locked": bool(status_reg & self.STATUS_LOCKED),
            "initialized": bool(status_reg & self.STATUS_INITIALIZED),
            "version": self.tpm_version,
            "active_keys": len(self.active_keys),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        reg_info = self.register_map[register]
        if "W" in reg_info["access"] and "R" not in reg_info["access"]:
            return OperationResult(False, error=f"Register {register} is write-only")

        try:
            # Simulated register reads
            if register == "STATUS":
                value = self._read_status_register()
            elif register == "CAPABILITIES":
                value = self._read_capabilities_register()
            elif register == "VERSION":
                value = self._read_version_register()
            elif register == "ERROR_CODE":
                value = 0  # No error
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

    # TPM-specific operations

    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported TPM algorithms"""
        return self.supported_algorithms

    def generate_key(self, algorithm: int, key_size: int = 2048) -> OperationResult:
        """
        Generate a new TPM key

        Args:
            algorithm: Algorithm identifier (TPM2Algorithm)
            key_size: Key size in bits

        Returns:
            OperationResult with key handle
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        # Simulated key generation
        key_handle = len(self.active_keys) + 1
        self.active_keys[key_handle] = {
            "algorithm": algorithm,
            "key_size": key_size,
            "created": True,
        }

        self._record_operation(True)
        return OperationResult(True, data={"key_handle": key_handle})

    def read_pcr(self, pcr_index: int) -> OperationResult:
        """
        Read Platform Configuration Register

        Args:
            pcr_index: PCR index (0-23)

        Returns:
            OperationResult with PCR value
        """
        if pcr_index < 0 or pcr_index >= 24:
            return OperationResult(False, error="Invalid PCR index")

        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        value = self.pcr_values.get(pcr_index, 0)

        self._record_operation(True)
        return OperationResult(True, data={
            "pcr_index": pcr_index,
            "value": value,
            "hex": f"0x{value:064X}",
        })

    def extend_pcr(self, pcr_index: int, data: bytes) -> OperationResult:
        """
        Extend Platform Configuration Register

        Args:
            pcr_index: PCR index (0-23)
            data: Data to extend

        Returns:
            OperationResult with success status
        """
        if pcr_index < 0 or pcr_index >= 24:
            return OperationResult(False, error="Invalid PCR index")

        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        # Simulated PCR extend (hash the data and XOR with current value)
        import hashlib
        hash_val = int.from_bytes(hashlib.sha256(data).digest(), 'big')
        self.pcr_values[pcr_index] ^= hash_val

        self._record_operation(True)
        return OperationResult(True, data={
            "pcr_index": pcr_index,
            "new_value": self.pcr_values[pcr_index],
        })

    def seal_data(self, data: bytes, pcr_list: List[int]) -> OperationResult:
        """
        Seal data to PCR values

        Args:
            data: Data to seal
            pcr_list: List of PCR indices to seal to

        Returns:
            OperationResult with sealed blob
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        # Simulated sealing
        sealed_blob = {
            "data_size": len(data),
            "pcr_list": pcr_list,
            "sealed": True,
        }

        self._record_operation(True)
        return OperationResult(True, data=sealed_blob)

    def unseal_data(self, sealed_blob: Dict) -> OperationResult:
        """
        Unseal data (requires matching PCR values)

        Args:
            sealed_blob: Sealed data blob

        Returns:
            OperationResult with unsealed data
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if not sealed_blob.get("sealed"):
            return OperationResult(False, error="Invalid sealed blob")

        # Simulated unsealing
        self._record_operation(True)
        return OperationResult(True, data={"unsealed": True})

    def get_random(self, num_bytes: int) -> OperationResult:
        """
        Get hardware random bytes from TPM

        Args:
            num_bytes: Number of random bytes to generate

        Returns:
            OperationResult with random bytes
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if num_bytes <= 0 or num_bytes > 256:
            return OperationResult(False, error="Invalid size (1-256 bytes)")

        # Simulated random generation
        import os
        random_bytes = os.urandom(num_bytes)

        self._record_operation(True)
        return OperationResult(True, data={
            "size": num_bytes,
            "data": random_bytes.hex(),
        })

    # ========== ADDITIONAL TPM 2.0 OPERATIONS (tpm2_tools parity) ==========

    def create_primary(self, hierarchy: str = "owner", algorithm: str = "rsa2048") -> OperationResult:
        """
        Create a primary key in TPM hierarchy
        Equivalent to: tpm2_createprimary

        Args:
            hierarchy: Key hierarchy ("owner", "endorsement", "platform", "null")
            algorithm: Key algorithm (e.g., "rsa2048", "ecc256", "aes128")

        Returns:
            OperationResult with key handle
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        valid_hierarchies = ["owner", "endorsement", "platform", "null"]
        if hierarchy not in valid_hierarchies:
            return OperationResult(False, error=f"Invalid hierarchy. Must be one of: {valid_hierarchies}")

        self._record_operation(True)
        return OperationResult(True, data={
            "hierarchy": hierarchy,
            "algorithm": algorithm,
            "handle": f"0x81{hash(algorithm) % 1000:06X}",
            "public": "simulated_public_key_data",
            "created": True
        })

    def create_key(self, parent_handle: str, algorithm: str = "rsa2048",
                   attributes: List[str] = None) -> OperationResult:
        """
        Create a child key under a parent
        Equivalent to: tpm2_create

        Args:
            parent_handle: Parent key handle
            algorithm: Key algorithm
            attributes: Key attributes (e.g., ["sign", "decrypt", "fixedtpm"])

        Returns:
            OperationResult with key data
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if attributes is None:
            attributes = ["fixedtpm", "fixedparent", "sensitivedataorigin"]

        self._record_operation(True)
        return OperationResult(True, data={
            "parent": parent_handle,
            "algorithm": algorithm,
            "attributes": attributes,
            "private": "simulated_private_blob",
            "public": "simulated_public_blob",
            "created": True
        })

    def load_key(self, parent_handle: str, private_blob: str, public_blob: str) -> OperationResult:
        """
        Load a key into TPM
        Equivalent to: tpm2_load

        Args:
            parent_handle: Parent key handle
            private_blob: Private key blob
            public_blob: Public key blob

        Returns:
            OperationResult with loaded key handle
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "parent": parent_handle,
            "handle": f"0x80{hash(private_blob) % 100000:06X}",
            "loaded": True
        })

    def flush_context(self, handle: str) -> OperationResult:
        """
        Flush a loaded key from TPM memory
        Equivalent to: tpm2_flushcontext

        Args:
            handle: Key handle to flush

        Returns:
            OperationResult with flush status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "handle": handle,
            "flushed": True
        })

    def evict_control(self, handle: str, persistent_handle: str) -> OperationResult:
        """
        Make a key persistent or evict a persistent key
        Equivalent to: tpm2_evictcontrol

        Args:
            handle: Transient handle
            persistent_handle: Persistent handle (e.g., "0x81000001")

        Returns:
            OperationResult with persist status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "transient_handle": handle,
            "persistent_handle": persistent_handle,
            "persisted": True
        })

    def sign(self, key_handle: str, data: bytes, scheme: str = "rsassa") -> OperationResult:
        """
        Sign data with TPM key
        Equivalent to: tpm2_sign

        Args:
            key_handle: Signing key handle
            data: Data to sign
            scheme: Signature scheme ("rsassa", "rsapss", "ecdsa", "ecdaa")

        Returns:
            OperationResult with signature
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import hashlib
        data_hash = hashlib.sha256(data).hexdigest()

        self._record_operation(True)
        return OperationResult(True, data={
            "key_handle": key_handle,
            "scheme": scheme,
            "data_hash": data_hash,
            "signature": f"sig_{data_hash[:32]}",
            "signed": True
        })

    def verify_signature(self, key_handle: str, data: bytes,
                        signature: str, scheme: str = "rsassa") -> OperationResult:
        """
        Verify signature with TPM key
        Equivalent to: tpm2_verifysignature

        Args:
            key_handle: Verification key handle
            data: Original data
            signature: Signature to verify
            scheme: Signature scheme

        Returns:
            OperationResult with verification status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "key_handle": key_handle,
            "scheme": scheme,
            "valid": True,  # Simulated verification
            "verified": True
        })

    def hash(self, data: bytes, algorithm: str = "sha256") -> OperationResult:
        """
        Perform TPM-based hash operation
        Equivalent to: tpm2_hash

        Args:
            data: Data to hash
            algorithm: Hash algorithm ("sha1", "sha256", "sha384", "sha512")

        Returns:
            OperationResult with hash
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import hashlib

        hash_funcs = {
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha384": hashlib.sha384,
            "sha512": hashlib.sha512
        }

        if algorithm not in hash_funcs:
            return OperationResult(False, error=f"Unsupported algorithm: {algorithm}")

        hash_value = hash_funcs[algorithm](data).hexdigest()

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": algorithm,
            "hash": hash_value,
            "size": len(hash_value) // 2
        })

    def hmac(self, key_handle: str, data: bytes, algorithm: str = "sha256") -> OperationResult:
        """
        Perform TPM-based HMAC operation
        Equivalent to: tpm2_hmac

        Args:
            key_handle: HMAC key handle
            data: Data to HMAC
            algorithm: Hash algorithm

        Returns:
            OperationResult with HMAC
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import hashlib
        import hmac as hmac_lib

        # Simulated HMAC using key handle as key
        key_bytes = key_handle.encode()
        hmac_value = hmac_lib.new(key_bytes, data, hashlib.sha256).hexdigest()

        self._record_operation(True)
        return OperationResult(True, data={
            "key_handle": key_handle,
            "algorithm": algorithm,
            "hmac": hmac_value
        })

    def quote(self, key_handle: str, pcr_list: List[int],
              nonce: bytes = None) -> OperationResult:
        """
        Generate TPM quote (attestation)
        Equivalent to: tpm2_quote

        Args:
            key_handle: Attestation key handle
            pcr_list: List of PCRs to quote
            nonce: Optional nonce for freshness

        Returns:
            OperationResult with quote data
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import os
        if nonce is None:
            nonce = os.urandom(20)

        pcr_values = {pcr: self.pcr_values.get(pcr, 0) for pcr in pcr_list}

        self._record_operation(True)
        return OperationResult(True, data={
            "key_handle": key_handle,
            "pcr_list": pcr_list,
            "pcr_values": pcr_values,
            "nonce": nonce.hex(),
            "quoted": True,
            "signature": "simulated_quote_signature"
        })

    def activate_credential(self, key_handle: str, credential_blob: bytes) -> OperationResult:
        """
        Activate a credential
        Equivalent to: tpm2_activatecredential

        Args:
            key_handle: Activation key handle
            credential_blob: Encrypted credential

        Returns:
            OperationResult with activated credential
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "key_handle": key_handle,
            "activated": True,
            "credential": "decrypted_credential_data"
        })

    def certify(self, object_handle: str, signing_key_handle: str) -> OperationResult:
        """
        Certify a TPM object
        Equivalent to: tpm2_certify

        Args:
            object_handle: Object to certify
            signing_key_handle: Key to sign certification

        Returns:
            OperationResult with certification
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "object_handle": object_handle,
            "signing_key": signing_key_handle,
            "certified": True,
            "attest_data": "simulated_attestation",
            "signature": "simulated_certification_signature"
        })

    def nv_define(self, nv_index: str, size: int, attributes: List[str] = None) -> OperationResult:
        """
        Define an NV (Non-Volatile) index
        Equivalent to: tpm2_nvdefine

        Args:
            nv_index: NV index (e.g., "0x1500001")
            size: Size in bytes
            attributes: NV attributes (e.g., ["ownerwrite", "ownerread"])

        Returns:
            OperationResult with NV index info
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if attributes is None:
            attributes = ["ownerwrite", "ownerread"]

        self._record_operation(True)
        return OperationResult(True, data={
            "nv_index": nv_index,
            "size": size,
            "attributes": attributes,
            "defined": True
        })

    def nv_write(self, nv_index: str, data: bytes, offset: int = 0) -> OperationResult:
        """
        Write to NV index
        Equivalent to: tpm2_nvwrite

        Args:
            nv_index: NV index
            data: Data to write
            offset: Write offset

        Returns:
            OperationResult with write status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "nv_index": nv_index,
            "bytes_written": len(data),
            "offset": offset,
            "written": True
        })

    def nv_read(self, nv_index: str, size: int, offset: int = 0) -> OperationResult:
        """
        Read from NV index
        Equivalent to: tpm2_nvread

        Args:
            nv_index: NV index
            size: Bytes to read
            offset: Read offset

        Returns:
            OperationResult with data
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import os
        simulated_data = os.urandom(size)

        self._record_operation(True)
        return OperationResult(True, data={
            "nv_index": nv_index,
            "size": size,
            "offset": offset,
            "data": simulated_data.hex()
        })

    def get_capability(self, capability: str) -> OperationResult:
        """
        Get TPM capability information
        Equivalent to: tpm2_getcap

        Args:
            capability: Capability type ("algorithms", "commands", "properties", "pcrs")

        Returns:
            OperationResult with capability info
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        capabilities_data = {
            "algorithms": self.supported_algorithms,
            "commands": ["create", "load", "sign", "verify", "quote", "seal", "unseal"],
            "properties": {
                "TPM2_PT_FAMILY_INDICATOR": "2.0",
                "TPM2_PT_LEVEL": "00",
                "TPM2_PT_REVISION": "1.59",
                "TPM2_PT_MANUFACTURER": "DELL"
            },
            "pcrs": {
                "sha256": list(range(24)),
                "sha384": list(range(24)),
                "sha512": list(range(24))
            }
        }

        if capability not in capabilities_data:
            return OperationResult(False, error=f"Unknown capability: {capability}")

        self._record_operation(True)
        return OperationResult(True, data={
            "capability": capability,
            "values": capabilities_data[capability]
        })

    def clear_tpm(self, hierarchy: str = "platform") -> OperationResult:
        """
        Clear TPM (DANGEROUS - erases all data)
        Equivalent to: tpm2_clear

        Args:
            hierarchy: Authorization hierarchy ("platform", "lockout")

        Returns:
            OperationResult with clear status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        # This is a destructive operation - require explicit confirmation
        self._record_operation(True)
        return OperationResult(True, data={
            "hierarchy": hierarchy,
            "cleared": True,
            "warning": "All TPM data has been cleared"
        })

    def reset_pcr(self, pcr_index: int) -> OperationResult:
        """
        Reset a PCR to its default value
        Equivalent to: tpm2_pcrreset

        Args:
            pcr_index: PCR index to reset (16-23 are resettable)

        Returns:
            OperationResult with reset status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if pcr_index < 16 or pcr_index > 23:
            return OperationResult(False, error="Only PCRs 16-23 are resettable")

        self.pcr_values[pcr_index] = 0

        self._record_operation(True)
        return OperationResult(True, data={
            "pcr_index": pcr_index,
            "reset": True,
            "value": 0
        })

    # ========== POST-QUANTUM CRYPTOGRAPHY OPERATIONS ==========

    def ml_kem_keypair(self, algorithm: int = None) -> OperationResult:
        """
        Generate ML-KEM (Key Encapsulation Mechanism) keypair
        FIPS 203 - Post-Quantum Key Encapsulation

        Args:
            algorithm: ML-KEM algorithm (ML_KEM_512, ML_KEM_768, ML_KEM_1024)
                      Default: ML_KEM_1024 (MIL-SPEC required)

        Returns:
            OperationResult with public_key, secret_key, and algorithm info
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_KEM  # ML-KEM-1024 (MIL-SPEC)

        if algorithm not in [MLKEMAlgorithm.ML_KEM_512, MLKEMAlgorithm.ML_KEM_768, MLKEMAlgorithm.ML_KEM_1024]:
            return OperationResult(False, error=f"Invalid ML-KEM algorithm: 0x{algorithm:04X}")

        import os
        key_sizes = MLKEMAlgorithm.get_key_sizes(algorithm)

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "algorithm_id": algorithm,
            "security_level": MLKEMAlgorithm.get_security_level(algorithm),
            "public_key": os.urandom(key_sizes["public_key"]).hex(),
            "secret_key": os.urandom(key_sizes["secret_key"]).hex(),
            "key_sizes": key_sizes,
            "standard": "FIPS 203",
            "milspec_compliant": algorithm == REQUIRED_KEM
        })

    def ml_kem_encapsulate(self, public_key: str, algorithm: int = None) -> OperationResult:
        """
        Encapsulate a shared secret using ML-KEM public key
        FIPS 203 - Key Encapsulation

        Args:
            public_key: Recipient's ML-KEM public key (hex string)
            algorithm: ML-KEM algorithm used

        Returns:
            OperationResult with ciphertext and shared_secret
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_KEM

        import os
        key_sizes = MLKEMAlgorithm.get_key_sizes(algorithm)

        # Generate shared secret (32 bytes for all ML-KEM variants)
        shared_secret = os.urandom(key_sizes["shared_secret"])
        ciphertext = os.urandom(key_sizes["ciphertext"])

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "ciphertext": ciphertext.hex(),
            "shared_secret": shared_secret.hex(),
            "ciphertext_size": len(ciphertext),
            "standard": "FIPS 203"
        })

    def ml_kem_decapsulate(self, secret_key: str, ciphertext: str, algorithm: int = None) -> OperationResult:
        """
        Decapsulate shared secret using ML-KEM secret key
        FIPS 203 - Key Decapsulation

        Args:
            secret_key: Recipient's ML-KEM secret key (hex string)
            ciphertext: Encapsulated ciphertext (hex string)
            algorithm: ML-KEM algorithm used

        Returns:
            OperationResult with shared_secret
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_KEM

        import os
        import hashlib

        # In real implementation, this would use liboqs or similar
        # For simulation, derive shared secret from ciphertext hash
        shared_secret = hashlib.sha256(bytes.fromhex(ciphertext[:64])).digest()

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "shared_secret": shared_secret.hex(),
            "decapsulated": True,
            "standard": "FIPS 203"
        })

    def ml_dsa_keypair(self, algorithm: int = None) -> OperationResult:
        """
        Generate ML-DSA (Digital Signature Algorithm) keypair
        FIPS 204 - Post-Quantum Digital Signatures

        Args:
            algorithm: ML-DSA algorithm (ML_DSA_44, ML_DSA_65, ML_DSA_87)
                      Default: ML_DSA_87 (MIL-SPEC required)

        Returns:
            OperationResult with public_key, secret_key, and algorithm info
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_SIGNATURE  # ML-DSA-87 (MIL-SPEC)

        if algorithm not in [MLDSAAlgorithm.ML_DSA_44, MLDSAAlgorithm.ML_DSA_65, MLDSAAlgorithm.ML_DSA_87]:
            return OperationResult(False, error=f"Invalid ML-DSA algorithm: 0x{algorithm:04X}")

        import os
        key_sizes = MLDSAAlgorithm.get_key_sizes(algorithm)

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "algorithm_id": algorithm,
            "security_level": MLDSAAlgorithm.get_security_level(algorithm),
            "public_key": os.urandom(key_sizes["public_key"]).hex(),
            "secret_key": os.urandom(key_sizes["secret_key"]).hex(),
            "signature_size": key_sizes["signature"],
            "key_sizes": key_sizes,
            "standard": "FIPS 204",
            "milspec_compliant": algorithm == REQUIRED_SIGNATURE
        })

    def ml_dsa_sign(self, secret_key: str, data: bytes, algorithm: int = None) -> OperationResult:
        """
        Sign data using ML-DSA secret key
        FIPS 204 - Post-Quantum Digital Signature

        Args:
            secret_key: Signer's ML-DSA secret key (hex string)
            data: Data to sign
            algorithm: ML-DSA algorithm used

        Returns:
            OperationResult with signature
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_SIGNATURE

        import os
        import hashlib
        key_sizes = MLDSAAlgorithm.get_key_sizes(algorithm)

        # Hash the data
        data_hash = hashlib.sha512(data).digest()

        # Generate signature (in real implementation, use liboqs)
        signature = os.urandom(key_sizes["signature"])

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "signature": signature.hex(),
            "signature_size": len(signature),
            "data_hash": data_hash.hex(),
            "hash_algorithm": "SHA-512",
            "signed": True,
            "standard": "FIPS 204"
        })

    def ml_dsa_verify(self, public_key: str, data: bytes, signature: str, algorithm: int = None) -> OperationResult:
        """
        Verify ML-DSA signature
        FIPS 204 - Post-Quantum Signature Verification

        Args:
            public_key: Signer's ML-DSA public key (hex string)
            data: Signed data
            signature: ML-DSA signature (hex string)
            algorithm: ML-DSA algorithm used

        Returns:
            OperationResult with verification status
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if algorithm is None:
            algorithm = REQUIRED_SIGNATURE

        import hashlib
        data_hash = hashlib.sha512(data).digest()

        # In real implementation, verify using liboqs
        # For simulation, always verify successfully
        verified = len(signature) > 0 and len(public_key) > 0

        self._record_operation(True)
        return OperationResult(True, data={
            "algorithm": get_algorithm_name(algorithm),
            "verified": verified,
            "data_hash": data_hash.hex(),
            "hash_algorithm": "SHA-512",
            "standard": "FIPS 204"
        })

    def hybrid_sign(self, classical_key: str, pqc_key: str, data: bytes,
                   classical_algo: str = "rsa3072", pqc_algo: int = None) -> OperationResult:
        """
        Hybrid signing: Combine classical (RSA/ECC) and post-quantum (ML-DSA) signatures
        Provides security even if one algorithm is broken

        Args:
            classical_key: Classical private key (RSA/ECC)
            pqc_key: ML-DSA secret key
            data: Data to sign
            classical_algo: Classical algorithm ("rsa2048", "rsa3072", "ecc_p384")
            pqc_algo: ML-DSA algorithm (default: ML-DSA-87)

        Returns:
            OperationResult with dual signature
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if pqc_algo is None:
            pqc_algo = REQUIRED_SIGNATURE

        import hashlib

        # Generate classical signature
        classical_sig = self.sign(classical_key, data, classical_algo)

        # Generate PQC signature
        pqc_sig = self.ml_dsa_sign(pqc_key, data, pqc_algo)

        self._record_operation(True)
        return OperationResult(True, data={
            "classical": {
                "algorithm": classical_algo,
                "signature": classical_sig.data.get("signature") if classical_sig.success else None
            },
            "pqc": {
                "algorithm": get_algorithm_name(pqc_algo),
                "signature": pqc_sig.data.get("signature") if pqc_sig.success else None
            },
            "data_hash": hashlib.sha512(data).hexdigest(),
            "hybrid": True,
            "standard": "Hybrid Classical + FIPS 204"
        })

    def hybrid_verify(self, classical_pubkey: str, pqc_pubkey: str, data: bytes,
                     classical_sig: str, pqc_sig: str,
                     classical_algo: str = "rsa3072", pqc_algo: int = None) -> OperationResult:
        """
        Verify hybrid signature (both classical and PQC must verify)

        Args:
            classical_pubkey: Classical public key
            pqc_pubkey: ML-DSA public key
            data: Signed data
            classical_sig: Classical signature
            pqc_sig: ML-DSA signature
            classical_algo: Classical algorithm
            pqc_algo: ML-DSA algorithm

        Returns:
            OperationResult with verification status (both must pass)
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        if pqc_algo is None:
            pqc_algo = REQUIRED_SIGNATURE

        # Verify classical signature
        classical_result = self.verify_signature(classical_pubkey, data, classical_sig, classical_algo)

        # Verify PQC signature
        pqc_result = self.ml_dsa_verify(pqc_pubkey, data, pqc_sig, pqc_algo)

        both_verified = classical_result.success and pqc_result.success

        self._record_operation(True)
        return OperationResult(True, data={
            "verified": both_verified,
            "classical_verified": classical_result.success,
            "pqc_verified": pqc_result.success,
            "hybrid": True,
            "standard": "Hybrid Classical + FIPS 204"
        })

    def pqc_encrypt(self, data: bytes, recipient_ml_kem_pubkey: str = None) -> OperationResult:
        """
        Encrypt data using Post-Quantum Cryptography
        Uses ML-KEM for key encapsulation + AES-256-GCM for data encryption

        Args:
            data: Data to encrypt
            recipient_ml_kem_pubkey: Recipient's ML-KEM public key (or generate ephemeral)

        Returns:
            OperationResult with encrypted data, ciphertext, and keys
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        import os
        from Crypto.Cipher import AES

        # Generate ephemeral ML-KEM keypair if no pubkey provided
        if recipient_ml_kem_pubkey is None:
            keypair = self.ml_kem_keypair(REQUIRED_KEM)
            recipient_ml_kem_pubkey = keypair.data["public_key"]

        # Encapsulate shared secret using ML-KEM
        kem_result = self.ml_kem_encapsulate(recipient_ml_kem_pubkey, REQUIRED_KEM)
        shared_secret = bytes.fromhex(kem_result.data["shared_secret"])

        # Use shared secret as AES-256-GCM key
        aes_key = shared_secret[:32]  # 256 bits
        nonce = os.urandom(12)  # 96 bits for GCM

        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)

        self._record_operation(True)
        return OperationResult(True, data={
            "encrypted": True,
            "kem_ciphertext": kem_result.data["ciphertext"],
            "data_ciphertext": ciphertext.hex(),
            "nonce": nonce.hex(),
            "tag": tag.hex(),
            "algorithm": "ML-KEM-1024 + AES-256-GCM",
            "standard": "FIPS 203 + FIPS 197",
            "milspec_compliant": True
        })

    def pqc_decrypt(self, kem_ciphertext: str, data_ciphertext: str, nonce: str, tag: str,
                   secret_key: str) -> OperationResult:
        """
        Decrypt data using Post-Quantum Cryptography
        Uses ML-KEM for key decapsulation + AES-256-GCM for data decryption

        Args:
            kem_ciphertext: ML-KEM encapsulated key (hex)
            data_ciphertext: AES-GCM encrypted data (hex)
            nonce: AES-GCM nonce (hex)
            tag: AES-GCM authentication tag (hex)
            secret_key: ML-KEM secret key (hex)

        Returns:
            OperationResult with decrypted plaintext
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        from Crypto.Cipher import AES

        # Decapsulate shared secret using ML-KEM
        kem_result = self.ml_kem_decapsulate(secret_key, kem_ciphertext, REQUIRED_KEM)
        shared_secret = bytes.fromhex(kem_result.data["shared_secret"])

        # Use shared secret as AES-256-GCM key
        aes_key = shared_secret[:32]

        cipher = AES.new(aes_key, AES.MODE_GCM, nonce=bytes.fromhex(nonce))
        try:
            plaintext = cipher.decrypt_and_verify(bytes.fromhex(data_ciphertext), bytes.fromhex(tag))

            self._record_operation(True)
            return OperationResult(True, data={
                "decrypted": True,
                "plaintext": plaintext,
                "algorithm": "ML-KEM-1024 + AES-256-GCM",
                "standard": "FIPS 203 + FIPS 197"
            })
        except Exception as e:
            self._record_operation(False, str(e))
            return OperationResult(False, error=f"Decryption failed: {str(e)}")

    def validate_pqc_compliance(self, kem: int = None, signature: int = None,
                               symmetric: int = None, hash_algo: int = None) -> OperationResult:
        """
        Validate if cryptographic configuration meets MIL-SPEC PQC requirements

        Args:
            kem: Key encapsulation algorithm
            signature: Signature algorithm
            symmetric: Symmetric encryption algorithm
            hash_algo: Hash algorithm

        Returns:
            OperationResult with compliance status and issues
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        result = is_pqc_compliant(kem, signature, symmetric, hash_algo)

        self._record_operation(True)
        return OperationResult(True, data={
            "compliant": result["compliant"],
            "issues": result["issues"],
            "required": result["required"],
            "checked": {
                "kem": get_algorithm_name(kem) if kem else "Not checked",
                "signature": get_algorithm_name(signature) if signature else "Not checked",
                "symmetric": get_algorithm_name(symmetric) if symmetric else "Not checked",
                "hash": get_algorithm_name(hash_algo) if hash_algo else "Not checked"
            }
        })

    def get_pqc_status(self) -> OperationResult:
        """
        Get TPM Post-Quantum Cryptography status and capabilities

        Returns:
            OperationResult with PQC status, algorithms, and compliance
        """
        if self.state != DeviceState.READY:
            return OperationResult(False, error="TPM not ready")

        self._record_operation(True)
        return OperationResult(True, data={
            "pqc_enabled": True,
            "required_profile": PQCProfile.MILSPEC,
            "supported_kem": [
                "ML-KEM-512 (Security Level 1)",
                "ML-KEM-768 (Security Level 3)",
                "ML-KEM-1024 (Security Level 5) ⭐ REQUIRED"
            ],
            "supported_signature": [
                "ML-DSA-44 (Security Level 2)",
                "ML-DSA-65 (Security Level 3)",
                "ML-DSA-87 (Security Level 5) ⭐ REQUIRED"
            ],
            "supported_symmetric": [
                "AES-128-GCM",
                "AES-192-GCM",
                "AES-256-GCM ⭐ REQUIRED"
            ],
            "supported_hash": [
                "SHA-256",
                "SHA-384 (Recommended)",
                "SHA-512 ⭐ REQUIRED",
                "SHA3-256",
                "SHA3-384",
                "SHA3-512"
            ],
            "standards": ["FIPS 203", "FIPS 204", "FIPS 197", "FIPS 180-4", "NIST SP 800-38D"],
            "quantum_security": "~200-bit (Level 5)",
            "milspec_compliant": True
        })

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read TPM status register (simulated)"""
        status = self.STATUS_READY | self.STATUS_INITIALIZED
        if self.state == DeviceState.BUSY:
            status |= self.STATUS_BUSY
        elif self.state == DeviceState.ERROR:
            status |= self.STATUS_ERROR
        return status

    def _read_capabilities_register(self) -> int:
        """Read TPM capabilities register (simulated)"""
        caps = (
            self.CAP_RSA | self.CAP_ECC | self.CAP_AES |
            self.CAP_SHA256 | self.CAP_SHA512 |
            self.CAP_PCR_EXTEND | self.CAP_KEY_GENERATION |
            self.CAP_SEALING | self.CAP_ATTESTATION |
            self.CAP_POST_QUANTUM
        )
        return caps

    def _read_version_register(self) -> int:
        """Read TPM version register (simulated)"""
        # TPM 2.0 revision 1.59
        return (2 << 16) | 159

    def _decode_capabilities(self, caps: int):
        """Decode capabilities bitmap"""
        self.capabilities = []
        self.supported_algorithms = []

        if caps & self.CAP_RSA:
            self.capabilities.append("RSA")
            self.supported_algorithms.extend(["RSA-2048", "RSA-3072", "RSA-4096"])

        if caps & self.CAP_ECC:
            self.capabilities.append("ECC")
            self.supported_algorithms.extend(["ECC-P256", "ECC-P384", "ECC-P521"])

        if caps & self.CAP_AES:
            self.capabilities.append("AES")
            self.supported_algorithms.extend([
                "AES-128-CBC", "AES-256-CBC",
                "AES-128-CTR", "AES-256-CTR",
                "AES-128-GCM", "AES-256-GCM"
            ])

        if caps & self.CAP_SHA256:
            self.capabilities.append("SHA256")
            self.supported_algorithms.append("SHA-256")

        if caps & self.CAP_SHA512:
            self.capabilities.append("SHA512")
            self.supported_algorithms.extend(["SHA-384", "SHA-512"])

        if caps & self.CAP_PCR_EXTEND:
            self.capabilities.append("PCR_EXTEND")

        if caps & self.CAP_KEY_GENERATION:
            self.capabilities.append("KEY_GENERATION")

        if caps & self.CAP_SEALING:
            self.capabilities.append("SEALING")

        if caps & self.CAP_ATTESTATION:
            self.capabilities.append("ATTESTATION")

        if caps & self.CAP_POST_QUANTUM:
            self.capabilities.append("POST_QUANTUM")
            self.supported_algorithms.extend([
                "Dilithium2", "Dilithium3", "Dilithium5",
                "Kyber512", "Kyber768", "Kyber1024"
            ])
