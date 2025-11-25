"""
Post-Quantum Cryptography Constants for DSMIL Framework

This module defines NIST-standardized post-quantum cryptographic algorithms
as mandated for MIL-SPEC systems. All DSMIL devices with encryption capabilities
must comply with these standards.

Standards Compliance:
- FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM)
- FIPS 204: Module-Lattice-Based Digital Signature Algorithm (ML-DSA)
- FIPS 197: Advanced Encryption Standard (AES)
- NIST SP 800-38D: Galois/Counter Mode (GCM)

Security Requirements:
- Key Encapsulation: ML-KEM-1024 (formerly CRYSTALS-KYBER Level 5)
- Digital Signatures: ML-DSA-87 (formerly CRYSTALS-DILITHIUM Level 5)
- Symmetric Encryption: AES-256-GCM
- Hash Functions: SHA-384 or SHA-512 (quantum-resistant)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 2.0.0 (Post-Quantum)
"""


class MLKEMAlgorithm:
    """
    ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism)
    FIPS 203 - Standardized Post-Quantum Key Encapsulation

    Formerly known as CRYSTALS-KYBER
    """

    # ML-KEM Parameter Sets (FIPS 203)
    ML_KEM_512 = 0x1000   # Security Level 1 (128-bit classical, ~100-bit quantum)
    ML_KEM_768 = 0x1001   # Security Level 3 (192-bit classical, ~150-bit quantum)
    ML_KEM_1024 = 0x1002  # Security Level 5 (256-bit classical, ~200-bit quantum) ⭐ REQUIRED

    # Legacy names (deprecated - for backward compatibility)
    KYBER512 = ML_KEM_512
    KYBER768 = ML_KEM_768
    KYBER1024 = ML_KEM_1024

    @staticmethod
    def get_name(algorithm_id: int) -> str:
        """Get human-readable name for ML-KEM algorithm"""
        names = {
            0x1000: "ML-KEM-512",
            0x1001: "ML-KEM-768",
            0x1002: "ML-KEM-1024",
        }
        return names.get(algorithm_id, f"Unknown ML-KEM (0x{algorithm_id:04X})")

    @staticmethod
    def get_security_level(algorithm_id: int) -> int:
        """Get NIST security level (1, 3, or 5)"""
        levels = {
            0x1000: 1,
            0x1001: 3,
            0x1002: 5,
        }
        return levels.get(algorithm_id, 0)

    @staticmethod
    def get_key_sizes(algorithm_id: int) -> dict:
        """Get key sizes in bytes for ML-KEM algorithm"""
        sizes = {
            0x1000: {"public_key": 800, "secret_key": 1632, "ciphertext": 768, "shared_secret": 32},
            0x1001: {"public_key": 1184, "secret_key": 2400, "ciphertext": 1088, "shared_secret": 32},
            0x1002: {"public_key": 1568, "secret_key": 3168, "ciphertext": 1568, "shared_secret": 32},
        }
        return sizes.get(algorithm_id, {})


class MLDSAAlgorithm:
    """
    ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
    FIPS 204 - Standardized Post-Quantum Digital Signatures

    Formerly known as CRYSTALS-DILITHIUM
    """

    # ML-DSA Parameter Sets (FIPS 204)
    ML_DSA_44 = 0x2000   # Security Level 2 (128-bit classical) - Small signatures
    ML_DSA_65 = 0x2001   # Security Level 3 (192-bit classical) - Medium signatures
    ML_DSA_87 = 0x2002   # Security Level 5 (256-bit classical) - Large signatures ⭐ REQUIRED

    # Legacy names (deprecated - for backward compatibility)
    DILITHIUM2 = ML_DSA_44
    DILITHIUM3 = ML_DSA_65
    DILITHIUM5 = ML_DSA_87

    @staticmethod
    def get_name(algorithm_id: int) -> str:
        """Get human-readable name for ML-DSA algorithm"""
        names = {
            0x2000: "ML-DSA-44",
            0x2001: "ML-DSA-65",
            0x2002: "ML-DSA-87",
        }
        return names.get(algorithm_id, f"Unknown ML-DSA (0x{algorithm_id:04X})")

    @staticmethod
    def get_security_level(algorithm_id: int) -> int:
        """Get NIST security level (2, 3, or 5)"""
        levels = {
            0x2000: 2,
            0x2001: 3,
            0x2002: 5,
        }
        return levels.get(algorithm_id, 0)

    @staticmethod
    def get_key_sizes(algorithm_id: int) -> dict:
        """Get key sizes in bytes for ML-DSA algorithm"""
        sizes = {
            0x2000: {"public_key": 1312, "secret_key": 2528, "signature": 2420},
            0x2001: {"public_key": 1952, "secret_key": 4000, "signature": 3293},
            0x2002: {"public_key": 2592, "secret_key": 4864, "signature": 4595},
        }
        return sizes.get(algorithm_id, {})


class AESAlgorithm:
    """
    AES (Advanced Encryption Standard) with GCM Mode
    FIPS 197 + NIST SP 800-38D

    Symmetric encryption for data protection
    """

    # AES Key Sizes with GCM mode
    AES_128_GCM = 0x3000  # 128-bit AES-GCM
    AES_192_GCM = 0x3001  # 192-bit AES-GCM
    AES_256_GCM = 0x3002  # 256-bit AES-GCM ⭐ REQUIRED

    # Other AES modes (for compatibility)
    AES_128_CBC = 0x3010  # 128-bit AES-CBC (deprecated for new systems)
    AES_256_CBC = 0x3011  # 256-bit AES-CBC (deprecated for new systems)
    AES_128_CTR = 0x3020  # 128-bit AES-CTR
    AES_256_CTR = 0x3021  # 256-bit AES-CTR

    @staticmethod
    def get_name(algorithm_id: int) -> str:
        """Get human-readable name for AES algorithm"""
        names = {
            0x3000: "AES-128-GCM",
            0x3001: "AES-192-GCM",
            0x3002: "AES-256-GCM",
            0x3010: "AES-128-CBC",
            0x3011: "AES-256-CBC",
            0x3020: "AES-128-CTR",
            0x3021: "AES-256-CTR",
        }
        return names.get(algorithm_id, f"Unknown AES (0x{algorithm_id:04X})")

    @staticmethod
    def get_key_size(algorithm_id: int) -> int:
        """Get key size in bits"""
        sizes = {
            0x3000: 128,
            0x3001: 192,
            0x3002: 256,
            0x3010: 128,
            0x3011: 256,
            0x3020: 128,
            0x3021: 256,
        }
        return sizes.get(algorithm_id, 0)

    @staticmethod
    def is_gcm(algorithm_id: int) -> bool:
        """Check if algorithm uses GCM mode"""
        return algorithm_id in [0x3000, 0x3001, 0x3002]


class HashAlgorithm:
    """
    Cryptographic Hash Functions
    Post-quantum secure hash functions
    """

    # SHA-2 Family (quantum-resistant with increased output sizes)
    SHA256 = 0x4000    # 256-bit output (adequate but not recommended for PQC)
    SHA384 = 0x4001    # 384-bit output (recommended for PQC) ⭐ RECOMMENDED
    SHA512 = 0x4002    # 512-bit output (recommended for PQC) ⭐ RECOMMENDED

    # SHA-3 Family (quantum-resistant)
    SHA3_256 = 0x4010  # 256-bit output
    SHA3_384 = 0x4011  # 384-bit output
    SHA3_512 = 0x4012  # 512-bit output

    # SHAKE (Extendable-Output Functions)
    SHAKE128 = 0x4020  # Variable output
    SHAKE256 = 0x4021  # Variable output

    @staticmethod
    def get_name(algorithm_id: int) -> str:
        """Get human-readable name for hash algorithm"""
        names = {
            0x4000: "SHA-256",
            0x4001: "SHA-384",
            0x4002: "SHA-512",
            0x4010: "SHA3-256",
            0x4011: "SHA3-384",
            0x4012: "SHA3-512",
            0x4020: "SHAKE128",
            0x4021: "SHAKE256",
        }
        return names.get(algorithm_id, f"Unknown Hash (0x{algorithm_id:04X})")

    @staticmethod
    def get_output_size(algorithm_id: int) -> int:
        """Get hash output size in bits"""
        sizes = {
            0x4000: 256,
            0x4001: 384,
            0x4002: 512,
            0x4010: 256,
            0x4011: 384,
            0x4012: 512,
        }
        return sizes.get(algorithm_id, 0)


class PQCProfile:
    """
    Post-Quantum Cryptography Compliance Profiles

    Defines standardized algorithm combinations for different security requirements
    """

    # MIL-SPEC Compliance Profile (REQUIRED for military systems)
    MILSPEC = {
        "name": "MIL-SPEC Post-Quantum",
        "kem": MLKEMAlgorithm.ML_KEM_1024,
        "signature": MLDSAAlgorithm.ML_DSA_87,
        "symmetric": AESAlgorithm.AES_256_GCM,
        "hash": HashAlgorithm.SHA512,
        "security_level": 5,
        "quantum_security": "~200-bit",
        "compliance": ["FIPS 203", "FIPS 204", "FIPS 197", "NIST SP 800-38D"],
    }

    # High Security Profile (recommended for sensitive data)
    HIGH_SECURITY = {
        "name": "High Security",
        "kem": MLKEMAlgorithm.ML_KEM_1024,
        "signature": MLDSAAlgorithm.ML_DSA_87,
        "symmetric": AESAlgorithm.AES_256_GCM,
        "hash": HashAlgorithm.SHA384,
        "security_level": 5,
        "quantum_security": "~200-bit",
    }

    # Standard Profile (balanced security/performance)
    STANDARD = {
        "name": "Standard Security",
        "kem": MLKEMAlgorithm.ML_KEM_768,
        "signature": MLDSAAlgorithm.ML_DSA_65,
        "symmetric": AESAlgorithm.AES_256_GCM,
        "hash": HashAlgorithm.SHA384,
        "security_level": 3,
        "quantum_security": "~150-bit",
    }

    # Lightweight Profile (IoT/embedded systems)
    LIGHTWEIGHT = {
        "name": "Lightweight Security",
        "kem": MLKEMAlgorithm.ML_KEM_512,
        "signature": MLDSAAlgorithm.ML_DSA_44,
        "symmetric": AESAlgorithm.AES_128_GCM,
        "hash": HashAlgorithm.SHA256,
        "security_level": 1,
        "quantum_security": "~100-bit",
    }

    @staticmethod
    def get_required_profile():
        """Get the required profile for DSMIL systems (MIL-SPEC)"""
        return PQCProfile.MILSPEC

    @staticmethod
    def validate_profile(kem, signature, symmetric, hash_algo):
        """Validate if a combination meets MIL-SPEC requirements"""
        milspec = PQCProfile.MILSPEC
        return (
            kem == milspec["kem"] and
            signature == milspec["signature"] and
            symmetric == milspec["symmetric"] and
            hash_algo == milspec["hash"]
        )


# Required algorithm combination for DSMIL systems
REQUIRED_KEM = MLKEMAlgorithm.ML_KEM_1024
REQUIRED_SIGNATURE = MLDSAAlgorithm.ML_DSA_87
REQUIRED_SYMMETRIC = AESAlgorithm.AES_256_GCM
REQUIRED_HASH = HashAlgorithm.SHA512

# Compliance statement
COMPLIANCE_STATEMENT = """
DSMIL Post-Quantum Cryptography Compliance:
- Key Encapsulation: ML-KEM-1024 (FIPS 203)
- Digital Signatures: ML-DSA-87 (FIPS 204)
- Symmetric Encryption: AES-256-GCM (FIPS 197 + SP 800-38D)
- Hash Functions: SHA-512 (FIPS 180-4)
- Security Level: 5 (256-bit classical, ~200-bit quantum)
- Quantum Security: Resistant to Shor's and Grover's algorithms
"""


def get_algorithm_name(algorithm_id: int) -> str:
    """Get human-readable name for any algorithm ID"""
    if 0x1000 <= algorithm_id <= 0x1FFF:
        return MLKEMAlgorithm.get_name(algorithm_id)
    elif 0x2000 <= algorithm_id <= 0x2FFF:
        return MLDSAAlgorithm.get_name(algorithm_id)
    elif 0x3000 <= algorithm_id <= 0x3FFF:
        return AESAlgorithm.get_name(algorithm_id)
    elif 0x4000 <= algorithm_id <= 0x4FFF:
        return HashAlgorithm.get_name(algorithm_id)
    else:
        return f"Unknown Algorithm (0x{algorithm_id:04X})"


def is_pqc_compliant(kem=None, signature=None, symmetric=None, hash_algo=None) -> dict:
    """
    Check if algorithm selection is PQC compliant

    Returns:
        dict with 'compliant' bool and 'issues' list
    """
    issues = []

    if kem is not None and kem != REQUIRED_KEM:
        issues.append(f"KEM must be ML-KEM-1024, got {get_algorithm_name(kem)}")

    if signature is not None and signature != REQUIRED_SIGNATURE:
        issues.append(f"Signature must be ML-DSA-87, got {get_algorithm_name(signature)}")

    if symmetric is not None and symmetric != REQUIRED_SYMMETRIC:
        issues.append(f"Symmetric must be AES-256-GCM, got {get_algorithm_name(symmetric)}")

    if hash_algo is not None and hash_algo != REQUIRED_HASH:
        issues.append(f"Hash should be SHA-512, got {get_algorithm_name(hash_algo)}")

    return {
        "compliant": len(issues) == 0,
        "issues": issues,
        "required": PQCProfile.MILSPEC,
    }
