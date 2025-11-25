/**
 * TPM2 Compatibility Layer - High-Performance C Library
 * Military-grade acceleration for critical TPM operations
 *
 * Author: C-INTERNAL Agent
 * Date: 2025-09-23
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 *
 * SECURITY WARNING: This library contains classified cryptographic implementations
 * and hardware acceleration routines. Use only with proper authorization.
 */

#ifndef TPM2_COMPAT_ACCELERATED_H
#define TPM2_COMPAT_ACCELERATED_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>

/* =============================================================================
 * LIBRARY VERSION AND CONFIGURATION
 * =============================================================================
 */

#define TPM2_COMPAT_VERSION_MAJOR 1
#define TPM2_COMPAT_VERSION_MINOR 0
#define TPM2_COMPAT_VERSION_PATCH 0
#define TPM2_COMPAT_VERSION "1.0.0"

/* Security classification levels */
typedef enum {
    SECURITY_UNCLASSIFIED = 0,
    SECURITY_CONFIDENTIAL = 1,
    SECURITY_SECRET = 2,
    SECURITY_TOP_SECRET = 3
} tpm2_security_level_t;

/* Hardware acceleration capabilities */
typedef enum {
    ACCEL_NONE = 0,
    ACCEL_NPU = 1,
    ACCEL_GNA = 2,
    ACCEL_AVX512 = 4,
    ACCEL_AES_NI = 8,
    ACCEL_RDRAND = 16,
    ACCEL_ALL = 0xFF
} tpm2_acceleration_flags_t;

/* PCR bank types */
typedef enum {
    PCR_BANK_SHA256 = 0,
    PCR_BANK_SHA384 = 1,
    PCR_BANK_SHA3_256 = 2,
    PCR_BANK_SHA3_384 = 3,
    PCR_BANK_SHA512 = 4,
    PCR_BANK_SM3 = 5,
    PCR_BANK_RESERVED = 6,
    PCR_BANK_EXTENDED = 7
} tpm2_pcr_bank_t;

/* Return codes */
typedef enum {
    TPM2_RC_SUCCESS = 0,
    TPM2_RC_FAILURE = 1,
    TPM2_RC_BAD_PARAMETER = 2,
    TPM2_RC_INSUFFICIENT_BUFFER = 3,
    TPM2_RC_NOT_SUPPORTED = 4,
    TPM2_RC_NOT_PERMITTED = 5,
    TPM2_RC_HARDWARE_FAILURE = 6,
    TPM2_RC_NOT_INITIALIZED = 7,
    TPM2_RC_MEMORY_ERROR = 8,
    TPM2_RC_CRYPTO_ERROR = 9,
    TPM2_RC_SECURITY_VIOLATION = 10,
    TPM2_RC_INVALID_STATE = 11
} tpm2_rc_t;

/* =============================================================================
 * PCR ADDRESS TRANSLATION (HIGH PERFORMANCE)
 * =============================================================================
 */

/**
 * High-performance PCR decimal to hex translation
 * Uses lookup tables and SIMD optimization for maximum speed
 */
tpm2_rc_t tpm2_pcr_decimal_to_hex_fast(
    uint32_t pcr_decimal,
    tpm2_pcr_bank_t bank,
    uint16_t *pcr_hex_out
);

/**
 * High-performance PCR hex to decimal translation
 * Optimized reverse lookup with memory-mapped tables
 */
tpm2_rc_t tpm2_pcr_hex_to_decimal_fast(
    uint16_t pcr_hex,
    uint32_t *pcr_decimal_out,
    tpm2_pcr_bank_t *bank_out
);

/**
 * Batch PCR translation for bulk operations
 * Uses vectorized operations for processing multiple PCRs
 */
tpm2_rc_t tpm2_pcr_translate_batch(
    const uint32_t *pcr_decimals,
    size_t count,
    tpm2_pcr_bank_t bank,
    uint16_t *pcr_hexs_out
);

/**
 * Validate PCR range with hardware acceleration
 */
tpm2_rc_t tpm2_pcr_validate_range_accel(
    uint32_t pcr,
    bool is_hex,
    tpm2_pcr_bank_t *bank_out
);

/* =============================================================================
 * ME COMMAND WRAPPING (OPTIMIZED)
 * =============================================================================
 */

/* ME session handle */
typedef struct tpm2_me_session_t* tpm2_me_session_handle_t;

/* ME command types */
typedef enum {
    ME_CMD_TPM_STARTUP = 0x01,
    ME_CMD_TPM_COMMAND = 0x02,
    ME_CMD_TPM_SHUTDOWN = 0x03,
    ME_CMD_TPM_RESET = 0x04,
    ME_CMD_SESSION_ESTABLISH = 0x10,
    ME_CMD_SESSION_CLOSE = 0x11,
    ME_CMD_HEARTBEAT = 0x20
} tpm2_me_command_type_t;

/* ME session configuration */
typedef struct {
    uint32_t session_id;
    tpm2_security_level_t security_level;
    uint32_t capabilities;
    uint64_t timeout_ms;
    bool enable_encryption;
    bool enable_compression;
} tpm2_me_session_config_t;

/**
 * Initialize ME interface with hardware detection
 */
tpm2_rc_t tpm2_me_interface_init(
    tpm2_acceleration_flags_t accel_flags
);

/**
 * Establish optimized ME session
 */
tpm2_rc_t tpm2_me_session_establish(
    const tpm2_me_session_config_t *config,
    tpm2_me_session_handle_t *session_out
);

/**
 * High-performance TPM command wrapping
 * Uses zero-copy operations and SIMD for maximum throughput
 */
tpm2_rc_t tpm2_me_wrap_command_fast(
    tpm2_me_session_handle_t session,
    const uint8_t *tpm_command,
    size_t tpm_command_size,
    uint8_t *wrapped_command_out,
    size_t *wrapped_command_size_inout
);

/**
 * High-performance ME response unwrapping
 * Optimized for minimal memory copies and validation
 */
tpm2_rc_t tpm2_me_unwrap_response_fast(
    tpm2_me_session_handle_t session,
    const uint8_t *me_response,
    size_t me_response_size,
    uint8_t *tpm_response_out,
    size_t *tpm_response_size_inout
);

/**
 * Send TPM command via ME with full optimization
 */
tpm2_rc_t tpm2_me_send_tpm_command(
    tpm2_me_session_handle_t session,
    const uint8_t *tpm_command,
    size_t tpm_command_size,
    uint8_t *tpm_response_out,
    size_t *tpm_response_size_inout,
    uint64_t timeout_ms
);

/**
 * Close ME session and cleanup
 */
tpm2_rc_t tpm2_me_session_close(
    tpm2_me_session_handle_t session
);

/**
 * Cleanup ME interface
 */
void tpm2_me_interface_cleanup(void);

/* =============================================================================
 * CRYPTOGRAPHIC ACCELERATION
 * =============================================================================
 */

/* Cryptographic algorithm types - Full TPM 2.0 Support */
typedef enum {
    /* ===== Hash Algorithms ===== */
    CRYPTO_ALG_SHA1 = 0,           /* Legacy SHA-1 (for compatibility) */
    CRYPTO_ALG_SHA256 = 1,
    CRYPTO_ALG_SHA384 = 2,
    CRYPTO_ALG_SHA512 = 3,
    CRYPTO_ALG_SHA3_256 = 4,
    CRYPTO_ALG_SHA3_384 = 5,
    CRYPTO_ALG_SHA3_512 = 6,       /* NEW: SHA3-512 */
    CRYPTO_ALG_SM3_256 = 7,        /* NEW: Chinese SM3 hash */
    CRYPTO_ALG_SHAKE128 = 8,       /* NEW: SHAKE-128 XOF */
    CRYPTO_ALG_SHAKE256 = 9,       /* NEW: SHAKE-256 XOF */

    /* ===== Symmetric Encryption - AES Block Cipher Modes ===== */
    CRYPTO_ALG_AES_128_ECB = 10,   /* NEW: AES-128 ECB mode */
    CRYPTO_ALG_AES_256_ECB = 11,   /* NEW: AES-256 ECB mode */
    CRYPTO_ALG_AES_128_CBC = 12,
    CRYPTO_ALG_AES_256_CBC = 13,
    CRYPTO_ALG_AES_128_CTR = 14,   /* NEW: AES-128 CTR mode */
    CRYPTO_ALG_AES_256_CTR = 15,   /* NEW: AES-256 CTR mode */
    CRYPTO_ALG_AES_128_OFB = 16,   /* NEW: AES-128 OFB mode */
    CRYPTO_ALG_AES_256_OFB = 17,   /* NEW: AES-256 OFB mode */
    CRYPTO_ALG_AES_128_CFB = 18,   /* NEW: AES-128 CFB mode */
    CRYPTO_ALG_AES_256_CFB = 19,   /* NEW: AES-256 CFB mode */
    CRYPTO_ALG_AES_128_GCM = 20,
    CRYPTO_ALG_AES_256_GCM = 21,
    CRYPTO_ALG_AES_128_CCM = 22,   /* NEW: AES-128 CCM mode (AEAD) */
    CRYPTO_ALG_AES_256_CCM = 23,   /* NEW: AES-256 CCM mode (AEAD) */
    CRYPTO_ALG_AES_128_XTS = 24,   /* NEW: AES-128 XTS (disk encryption) */
    CRYPTO_ALG_AES_256_XTS = 25,   /* NEW: AES-256 XTS (disk encryption) */

    /* ===== Symmetric Encryption - Other Ciphers ===== */
    CRYPTO_ALG_3DES_EDE = 26,      /* NEW: Triple DES (legacy) */
    CRYPTO_ALG_CAMELLIA_128 = 27,  /* NEW: Camellia-128 */
    CRYPTO_ALG_CAMELLIA_256 = 28,  /* NEW: Camellia-256 */
    CRYPTO_ALG_SM4_128 = 29,       /* NEW: Chinese SM4 cipher */
    CRYPTO_ALG_CHACHA20 = 30,      /* NEW: ChaCha20 stream cipher */
    CRYPTO_ALG_CHACHA20_POLY1305 = 31, /* NEW: ChaCha20-Poly1305 AEAD */

    /* ===== Asymmetric - RSA ===== */
    CRYPTO_ALG_RSA_1024 = 32,      /* NEW: RSA-1024 (legacy) */
    CRYPTO_ALG_RSA_2048 = 33,
    CRYPTO_ALG_RSA_3072 = 34,      /* NEW: RSA-3072 */
    CRYPTO_ALG_RSA_4096 = 35,
    CRYPTO_ALG_RSA_8192 = 36,      /* NEW: RSA-8192 (high security) */

    /* ===== Asymmetric - Elliptic Curve ===== */
    CRYPTO_ALG_ECC_P192 = 37,      /* NEW: NIST P-192 */
    CRYPTO_ALG_ECC_P224 = 38,      /* NEW: NIST P-224 */
    CRYPTO_ALG_ECC_P256 = 39,      /* NIST P-256 (secp256r1) */
    CRYPTO_ALG_ECC_P384 = 40,      /* NIST P-384 (secp384r1) */
    CRYPTO_ALG_ECC_P521 = 41,      /* NEW: NIST P-521 (secp521r1) */
    CRYPTO_ALG_ECC_SM2_P256 = 42,  /* NEW: Chinese SM2 curve */
    CRYPTO_ALG_ECC_BN_P256 = 43,   /* NEW: Barreto-Naehrig 256 */
    CRYPTO_ALG_ECC_BN_P638 = 44,   /* NEW: Barreto-Naehrig 638 */
    CRYPTO_ALG_ECC_CURVE25519 = 45,/* NEW: Curve25519 (X25519) */
    CRYPTO_ALG_ECC_CURVE448 = 46,  /* NEW: Curve448 (X448) */
    CRYPTO_ALG_ECC_ED25519 = 47,   /* NEW: Ed25519 signatures */
    CRYPTO_ALG_ECC_ED448 = 48,     /* NEW: Ed448 signatures */

    /* ===== HMAC Algorithms ===== */
    CRYPTO_ALG_HMAC_SHA1 = 49,     /* NEW: HMAC-SHA1 */
    CRYPTO_ALG_HMAC_SHA256 = 50,   /* NEW: HMAC-SHA256 */
    CRYPTO_ALG_HMAC_SHA384 = 51,   /* NEW: HMAC-SHA384 */
    CRYPTO_ALG_HMAC_SHA512 = 52,   /* NEW: HMAC-SHA512 */
    CRYPTO_ALG_HMAC_SM3 = 53,      /* NEW: HMAC-SM3 */

    /* ===== Key Derivation Functions ===== */
    CRYPTO_ALG_KDF_SP800_108 = 54, /* NEW: NIST SP800-108 KDF */
    CRYPTO_ALG_KDF_SP800_56A = 55, /* NEW: NIST SP800-56A KDF */
    CRYPTO_ALG_HKDF_SHA256 = 56,   /* NEW: HKDF with SHA-256 */
    CRYPTO_ALG_HKDF_SHA384 = 57,   /* NEW: HKDF with SHA-384 */
    CRYPTO_ALG_HKDF_SHA512 = 58,   /* NEW: HKDF with SHA-512 */
    CRYPTO_ALG_PBKDF2_SHA256 = 59, /* NEW: PBKDF2 with SHA-256 */
    CRYPTO_ALG_PBKDF2_SHA512 = 60, /* NEW: PBKDF2 with SHA-512 */
    CRYPTO_ALG_SCRYPT = 61,        /* NEW: scrypt KDF */
    CRYPTO_ALG_ARGON2I = 62,       /* NEW: Argon2i (memory-hard) */
    CRYPTO_ALG_ARGON2D = 63,       /* NEW: Argon2d (memory-hard) */
    CRYPTO_ALG_ARGON2ID = 64,      /* NEW: Argon2id (hybrid) */

    /* ===== Signature Schemes ===== */
    CRYPTO_ALG_RSA_SSA_PKCS1V15 = 65,  /* NEW: RSA signature PKCS#1 v1.5 */
    CRYPTO_ALG_RSA_PSS = 66,           /* NEW: RSA-PSS */
    CRYPTO_ALG_ECDSA_SHA256 = 67,      /* NEW: ECDSA with SHA-256 */
    CRYPTO_ALG_ECDSA_SHA384 = 68,      /* NEW: ECDSA with SHA-384 */
    CRYPTO_ALG_ECDSA_SHA512 = 69,      /* NEW: ECDSA with SHA-512 */
    CRYPTO_ALG_SCHNORR = 70,           /* NEW: Schnorr signatures */
    CRYPTO_ALG_SM2_SIGN = 71,          /* NEW: SM2 signature scheme */
    CRYPTO_ALG_ECDAA = 72,             /* NEW: Elliptic Curve DAA */

    /* ===== Key Agreement ===== */
    CRYPTO_ALG_ECDH = 73,          /* NEW: Elliptic Curve Diffie-Hellman */
    CRYPTO_ALG_ECMQV = 74,         /* NEW: Elliptic Curve MQV */
    CRYPTO_ALG_DH = 75,            /* NEW: Diffie-Hellman */

    /* ===== Mask Generation Functions ===== */
    CRYPTO_ALG_MGF1_SHA1 = 76,     /* NEW: MGF1 with SHA-1 */
    CRYPTO_ALG_MGF1_SHA256 = 77,   /* NEW: MGF1 with SHA-256 */
    CRYPTO_ALG_MGF1_SHA384 = 78,   /* NEW: MGF1 with SHA-384 */
    CRYPTO_ALG_MGF1_SHA512 = 79,   /* NEW: MGF1 with SHA-512 */

    /* ===== Post-Quantum Cryptography (Future) ===== */
    CRYPTO_ALG_KYBER512 = 80,      /* NEW: Kyber-512 (NIST PQC) */
    CRYPTO_ALG_KYBER768 = 81,      /* NEW: Kyber-768 (NIST PQC) */
    CRYPTO_ALG_KYBER1024 = 82,     /* NEW: Kyber-1024 (NIST PQC) */
    CRYPTO_ALG_DILITHIUM2 = 83,    /* NEW: Dilithium2 (NIST PQC) */
    CRYPTO_ALG_DILITHIUM3 = 84,    /* NEW: Dilithium3 (NIST PQC) */
    CRYPTO_ALG_DILITHIUM5 = 85,    /* NEW: Dilithium5 (NIST PQC) */
    CRYPTO_ALG_FALCON512 = 86,     /* NEW: Falcon-512 (NIST PQC) */
    CRYPTO_ALG_FALCON1024 = 87,    /* NEW: Falcon-1024 (NIST PQC) */

    CRYPTO_ALG_MAX = 88            /* Maximum algorithm ID */
} tpm2_crypto_algorithm_t;

/* Cryptographic context handle */
typedef struct tpm2_crypto_context_t* tpm2_crypto_context_handle_t;

/**
 * Initialize cryptographic acceleration
 */
tpm2_rc_t tpm2_crypto_init(
    tpm2_acceleration_flags_t accel_flags,
    tpm2_security_level_t min_security_level
);

/**
 * Create cryptographic context
 */
tpm2_rc_t tpm2_crypto_context_create(
    tpm2_crypto_algorithm_t algorithm,
    const uint8_t *key_material,
    size_t key_size,
    tpm2_crypto_context_handle_t *context_out
);

/**
 * Hardware-accelerated hash computation
 */
tpm2_rc_t tpm2_crypto_hash_accelerated(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *data,
    size_t data_size,
    uint8_t *hash_out,
    size_t *hash_size_inout
);

/**
 * Hardware-accelerated encryption
 */
tpm2_rc_t tpm2_crypto_encrypt_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *plaintext,
    size_t plaintext_size,
    const uint8_t *iv,
    size_t iv_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size_inout
);

/**
 * Hardware-accelerated decryption
 */
tpm2_rc_t tpm2_crypto_decrypt_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    const uint8_t *iv,
    size_t iv_size,
    uint8_t *plaintext_out,
    size_t *plaintext_size_inout
);

/**
 * Hardware-accelerated digital signature
 */
tpm2_rc_t tpm2_crypto_sign_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *data,
    size_t data_size,
    uint8_t *signature_out,
    size_t *signature_size_inout
);

/**
 * Hardware-accelerated signature verification
 */
tpm2_rc_t tpm2_crypto_verify_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *data,
    size_t data_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out
);

/**
 * Destroy cryptographic context
 */
tpm2_rc_t tpm2_crypto_context_destroy(
    tpm2_crypto_context_handle_t context
);

/* =============================================================================
 * HMAC OPERATIONS (NEW)
 * =============================================================================
 */

/**
 * Hardware-accelerated HMAC computation
 */
tpm2_rc_t tpm2_crypto_hmac_accelerated(
    tpm2_crypto_algorithm_t hmac_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *data,
    size_t data_size,
    uint8_t *hmac_out,
    size_t *hmac_size_inout
);

/**
 * Initialize incremental HMAC operation
 */
tpm2_rc_t tpm2_crypto_hmac_init(
    tpm2_crypto_context_handle_t *context_out,
    tpm2_crypto_algorithm_t hmac_alg,
    const uint8_t *key,
    size_t key_size
);

/**
 * Update HMAC with additional data
 */
tpm2_rc_t tpm2_crypto_hmac_update(
    tpm2_crypto_context_handle_t context,
    const uint8_t *data,
    size_t data_size
);

/**
 * Finalize HMAC computation
 */
tpm2_rc_t tpm2_crypto_hmac_final(
    tpm2_crypto_context_handle_t context,
    uint8_t *hmac_out,
    size_t *hmac_size_inout
);

/* =============================================================================
 * KEY DERIVATION FUNCTIONS (NEW)
 * =============================================================================
 */

/**
 * HKDF (HMAC-based Key Derivation Function)
 */
tpm2_rc_t tpm2_crypto_hkdf(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *salt,
    size_t salt_size,
    const uint8_t *input_key_material,
    size_t ikm_size,
    const uint8_t *info,
    size_t info_size,
    uint8_t *output_key_material,
    size_t okm_size
);

/**
 * PBKDF2 (Password-Based Key Derivation Function 2)
 */
tpm2_rc_t tpm2_crypto_pbkdf2(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint32_t iterations,
    uint8_t *derived_key,
    size_t key_size
);

/**
 * SP800-108 Counter Mode KDF
 */
tpm2_rc_t tpm2_crypto_kdf_sp800_108(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *key_derivation_key,
    size_t kdk_size,
    const uint8_t *label,
    size_t label_size,
    const uint8_t *context,
    size_t context_size,
    uint8_t *derived_key,
    size_t key_size
);

/**
 * scrypt - Memory-hard KDF
 */
tpm2_rc_t tpm2_crypto_scrypt(
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint64_t N,  /* CPU/memory cost parameter */
    uint32_t r,  /* Block size parameter */
    uint32_t p,  /* Parallelization parameter */
    uint8_t *derived_key,
    size_t key_size
);

/**
 * Argon2 - Memory-hard KDF (NIST recommended)
 */
tpm2_rc_t tpm2_crypto_argon2(
    tpm2_crypto_algorithm_t variant,  /* ARGON2I, ARGON2D, or ARGON2ID */
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint32_t time_cost,
    uint32_t memory_cost_kb,
    uint32_t parallelism,
    uint8_t *derived_key,
    size_t key_size
);

/* =============================================================================
 * KEY AGREEMENT PROTOCOLS (NEW)
 * =============================================================================
 */

/**
 * ECDH (Elliptic Curve Diffie-Hellman) - Compute shared secret
 */
tpm2_rc_t tpm2_crypto_ecdh(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *peer_public_key,
    size_t peer_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size
);

/**
 * ECDH - Generate ephemeral key pair
 */
tpm2_rc_t tpm2_crypto_ecdh_keygen(
    tpm2_crypto_algorithm_t curve,
    uint8_t *private_key_out,
    size_t *private_key_size,
    uint8_t *public_key_out,
    size_t *public_key_size
);

/**
 * Standard Diffie-Hellman
 */
tpm2_rc_t tpm2_crypto_dh(
    const uint8_t *prime,
    size_t prime_size,
    const uint8_t *generator,
    size_t generator_size,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *peer_public_key,
    size_t peer_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size
);

/**
 * EC-MQV - Elliptic Curve Menezes-Qu-Vanstone
 * Authenticated key agreement using static and ephemeral keys
 */
tpm2_rc_t tpm2_crypto_ecmqv(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *static_private_key,
    size_t static_private_key_size,
    const uint8_t *ephemeral_private_key,
    size_t ephemeral_private_key_size,
    const uint8_t *peer_static_public_key,
    size_t peer_static_public_key_size,
    const uint8_t *peer_ephemeral_public_key,
    size_t peer_ephemeral_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size
);

/**
 * EC-DAA Sign - Elliptic Curve Direct Anonymous Attestation
 * Anonymous signature for TPM attestation
 */
tpm2_rc_t tpm2_crypto_ecdaa_sign(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *daa_private_key,
    size_t daa_private_key_size,
    const uint8_t *basename,
    size_t basename_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size
);

/**
 * EC-DAA Verify - Verify anonymous attestation signature
 */
tpm2_rc_t tpm2_crypto_ecdaa_verify(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *daa_public_key,
    size_t daa_public_key_size,
    const uint8_t *basename,
    size_t basename_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out
);

/* =============================================================================
 * AEAD OPERATIONS (Authenticated Encryption with Associated Data) (NEW)
 * =============================================================================
 */

/**
 * AEAD Encryption (AES-GCM, AES-CCM, ChaCha20-Poly1305)
 */
tpm2_rc_t tpm2_crypto_aead_encrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *associated_data,
    size_t ad_size,
    const uint8_t *plaintext,
    size_t plaintext_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size,
    uint8_t *tag_out,
    size_t tag_size
);

/**
 * AEAD Decryption
 */
tpm2_rc_t tpm2_crypto_aead_decrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *associated_data,
    size_t ad_size,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    const uint8_t *tag,
    size_t tag_size,
    uint8_t *plaintext_out,
    size_t *plaintext_size
);

/* =============================================================================
 * EXTENDED SIGNATURE OPERATIONS (NEW)
 * =============================================================================
 */

/**
 * RSA-PSS Signature (Probabilistic Signature Scheme)
 */
tpm2_rc_t tpm2_crypto_rsa_pss_sign(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *message,
    size_t message_size,
    uint32_t salt_length,
    uint8_t *signature_out,
    size_t *signature_size
);

/**
 * RSA-PSS Verify
 */
tpm2_rc_t tpm2_crypto_rsa_pss_verify(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    uint32_t salt_length,
    bool *valid_out
);

/**
 * Ed25519 Signature
 */
tpm2_rc_t tpm2_crypto_ed25519_sign(
    const uint8_t *private_key,  /* 32 bytes */
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out       /* 64 bytes */
);

/**
 * Ed25519 Verify
 */
tpm2_rc_t tpm2_crypto_ed25519_verify(
    const uint8_t *public_key,   /* 32 bytes */
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,    /* 64 bytes */
    bool *valid_out
);

/**
 * Schnorr Signature
 */
tpm2_rc_t tpm2_crypto_schnorr_sign(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size
);

/**
 * Schnorr Verify
 */
tpm2_rc_t tpm2_crypto_schnorr_verify(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out
);

/**
 * MGF1 - Mask Generation Function 1 (PKCS#1 v2.1)
 * Used in RSA-OAEP and RSA-PSS
 */
tpm2_rc_t tpm2_crypto_mgf1(
    tpm2_crypto_algorithm_t hash_alg,  /* CRYPTO_ALG_MGF1_* */
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *mask,
    size_t mask_len
);

/* =============================================================================
 * POST-QUANTUM CRYPTOGRAPHY (NIST Winners)
 * =============================================================================
 */

/* Kyber - Post-Quantum KEM */
tpm2_rc_t tpm2_crypto_kyber_keygen(
    tpm2_crypto_algorithm_t kyber_variant,  /* KYBER512/768/1024 */
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size
);

tpm2_rc_t tpm2_crypto_kyber_encapsulate(
    tpm2_crypto_algorithm_t kyber_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size,
    uint8_t *shared_secret_out,
    size_t *shared_secret_size
);

tpm2_rc_t tpm2_crypto_kyber_decapsulate(
    tpm2_crypto_algorithm_t kyber_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    uint8_t *shared_secret_out,
    size_t *shared_secret_size
);

/* Dilithium - Post-Quantum Signatures */
tpm2_rc_t tpm2_crypto_dilithium_keygen(
    tpm2_crypto_algorithm_t dilithium_variant,  /* DILITHIUM2/3/5 */
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size
);

tpm2_rc_t tpm2_crypto_dilithium_sign(
    tpm2_crypto_algorithm_t dilithium_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size
);

tpm2_rc_t tpm2_crypto_dilithium_verify(
    tpm2_crypto_algorithm_t dilithium_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out
);

/* Falcon - Post-Quantum Signatures */
tpm2_rc_t tpm2_crypto_falcon_keygen(
    tpm2_crypto_algorithm_t falcon_variant,  /* FALCON512/1024 */
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size
);

tpm2_rc_t tpm2_crypto_falcon_sign(
    tpm2_crypto_algorithm_t falcon_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size
);

tpm2_rc_t tpm2_crypto_falcon_verify(
    tpm2_crypto_algorithm_t falcon_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out
);

/* =============================================================================
 * AGENT ATTESTATION API (Enhanced for TPM 2.0)
 * =============================================================================
 */

#define TPM2_AGENT_MAX_ID_LENGTH 64
#define TPM2_AGENT_MAX_ATTEST_BLOB 4096
#define TPM2_AGENT_MAX_SIGNATURE 512
#define TPM2_AGENT_MAX_PUBLIC_KEY 1024

/* Agent attestation structure */
typedef struct {
    char agent_id[TPM2_AGENT_MAX_ID_LENGTH];
    uint32_t pcr_index;
    uint16_t hash_algorithm;
    uint8_t task_nonce[64];
    size_t task_nonce_len;
    uint8_t pcr_digest[64];
    size_t pcr_digest_len;
    uint8_t attestation_blob[TPM2_AGENT_MAX_ATTEST_BLOB];
    size_t attestation_blob_len;
    uint8_t signature[TPM2_AGENT_MAX_SIGNATURE];
    size_t signature_len;
    uint8_t public_key[TPM2_AGENT_MAX_PUBLIC_KEY];
    size_t public_key_len;
} tpm2_agent_attestation_t;

/**
 * Begin TPM-attested agent task
 */
tpm2_rc_t tpm2_agent_task_begin(
    const char *agent_id,
    const uint8_t *task_descriptor,
    size_t descriptor_len
);

/**
 * Complete TPM-attested agent task and generate attestation
 */
tpm2_rc_t tpm2_agent_task_complete(
    const char *agent_id,
    const uint8_t *result_digest,
    size_t result_digest_len,
    tpm2_agent_attestation_t *attestation_out
);

/**
 * Verify TPM agent attestation
 */
tpm2_rc_t tpm2_agent_task_verify(
    const char *agent_id,
    const uint8_t *expected_result_digest,
    size_t result_digest_len,
    const tpm2_agent_attestation_t *attestation
);

/**
 * Cleanup cryptographic acceleration
 */
void tpm2_crypto_cleanup(void);

/* =============================================================================
 * MEMORY-MAPPED I/O FOR TPM DEVICE ACCESS
 * =============================================================================
 */

/* TPM device handle */
typedef struct tpm2_device_t* tpm2_device_handle_t;

/* TPM device configuration */
typedef struct {
    const char *device_path;
    uint32_t base_address;
    size_t memory_size;
    bool enable_interrupts;
    bool enable_dma;
    uint32_t timeout_ms;
} tpm2_device_config_t;

/**
 * Open TPM device with memory mapping
 */
tpm2_rc_t tpm2_device_open(
    const tpm2_device_config_t *config,
    tpm2_device_handle_t *device_out
);

/**
 * Direct memory-mapped register read
 */
tpm2_rc_t tpm2_device_read_register(
    tpm2_device_handle_t device,
    uint32_t offset,
    uint32_t *value_out
);

/**
 * Direct memory-mapped register write
 */
tpm2_rc_t tpm2_device_write_register(
    tpm2_device_handle_t device,
    uint32_t offset,
    uint32_t value
);

/**
 * High-performance bulk data transfer
 */
tpm2_rc_t tpm2_device_transfer_bulk(
    tpm2_device_handle_t device,
    const uint8_t *write_buffer,
    size_t write_size,
    uint8_t *read_buffer,
    size_t *read_size_inout,
    uint32_t timeout_ms
);

/**
 * Interrupt-driven I/O operation
 */
tpm2_rc_t tpm2_device_async_operation(
    tpm2_device_handle_t device,
    const uint8_t *command,
    size_t command_size,
    uint8_t *response_buffer,
    size_t response_buffer_size,
    void (*completion_callback)(tpm2_rc_t result, void *user_data),
    void *user_data
);

/**
 * Close TPM device and cleanup mapping
 */
tpm2_rc_t tpm2_device_close(
    tpm2_device_handle_t device
);

/* =============================================================================
 * INTEL NPU/GNA HARDWARE ACCELERATION
 * =============================================================================
 */

/* NPU context handle */
typedef struct tpm2_npu_context_t* tpm2_npu_context_handle_t;

/* NPU operation types */
typedef enum {
    NPU_OP_NEURAL_HASH = 0,
    NPU_OP_PATTERN_MATCH = 1,
    NPU_OP_ANOMALY_DETECT = 2,
    NPU_OP_CRYPTO_ACCEL = 3,
    NPU_OP_DATA_COMPRESS = 4
} tpm2_npu_operation_t;

/* NPU configuration */
typedef struct {
    uint32_t model_id;
    float performance_target_tops;
    uint32_t power_budget_mw;
    bool enable_quantization;
    bool enable_batch_processing;
} tpm2_npu_config_t;

/**
 * Initialize NPU acceleration
 */
tpm2_rc_t tpm2_npu_init(
    const tpm2_npu_config_t *config,
    tpm2_npu_context_handle_t *context_out
);

/**
 * Detect available NPU hardware
 */
tpm2_rc_t tpm2_npu_detect_hardware(
    uint32_t *model_id_out,
    float *tops_available_out,
    uint32_t *features_out
);

/**
 * NPU-accelerated cryptographic operations
 */
tpm2_rc_t tpm2_npu_crypto_operation(
    tpm2_npu_context_handle_t context,
    tpm2_npu_operation_t operation,
    const uint8_t *input_data,
    size_t input_size,
    uint8_t *output_data,
    size_t *output_size_inout
);

/**
 * NPU-accelerated anomaly detection for security
 */
tpm2_rc_t tpm2_npu_security_analysis(
    tpm2_npu_context_handle_t context,
    const uint8_t *tpm_command,
    size_t command_size,
    float *anomaly_score_out,
    bool *block_command_out
);

/**
 * Cleanup NPU acceleration
 */
tpm2_rc_t tpm2_npu_cleanup(
    tpm2_npu_context_handle_t context
);

/* =============================================================================
 * HARDWARE FAULT DETECTION AND RECOVERY
 * =============================================================================
 */

/* Fault types */
typedef enum {
    FAULT_NONE = 0,
    FAULT_MEMORY_ERROR = 1,
    FAULT_HARDWARE_FAILURE = 2,
    FAULT_SECURITY_VIOLATION = 3,
    FAULT_PERFORMANCE_DEGRADATION = 4,
    FAULT_THERMAL_EMERGENCY = 5
} tpm2_fault_type_t;

/* Fault severity levels */
typedef enum {
    SEVERITY_INFO = 0,
    SEVERITY_WARNING = 1,
    SEVERITY_ERROR = 2,
    SEVERITY_CRITICAL = 3,
    SEVERITY_EMERGENCY = 4
} tpm2_fault_severity_t;

/* Fault information structure */
typedef struct {
    tpm2_fault_type_t fault_type;
    tpm2_fault_severity_t severity;
    uint64_t timestamp;
    uint32_t error_code;
    char description[256];
    bool auto_recovery_possible;
} tpm2_fault_info_t;

/* Fault callback function type */
typedef void (*tpm2_fault_callback_t)(
    const tpm2_fault_info_t *fault_info,
    void *user_data
);

/**
 * Initialize fault detection system
 */
tpm2_rc_t tpm2_fault_detection_init(
    tpm2_fault_callback_t callback,
    void *user_data
);

/**
 * Enable hardware fault monitoring
 */
tpm2_rc_t tpm2_fault_monitoring_enable(
    tpm2_fault_type_t fault_types_mask
);

/**
 * Trigger fault recovery procedure
 */
tpm2_rc_t tpm2_fault_recovery_trigger(
    tpm2_fault_type_t fault_type,
    bool force_recovery
);

/**
 * Get current system health status
 */
tpm2_rc_t tpm2_get_system_health(
    float *health_score_out,
    tpm2_fault_info_t *active_faults,
    size_t *fault_count_inout
);

/**
 * Disable fault detection and cleanup
 */
void tpm2_fault_detection_cleanup(void);

/* =============================================================================
 * PERFORMANCE PROFILING AND OPTIMIZATION
 * =============================================================================
 */

/* Performance counter types */
typedef enum {
    PERF_COUNTER_TRANSLATIONS = 0,
    PERF_COUNTER_ME_COMMANDS = 1,
    PERF_COUNTER_CRYPTO_OPS = 2,
    PERF_COUNTER_DEVICE_IO = 3,
    PERF_COUNTER_NPU_OPS = 4,
    PERF_COUNTER_MEMORY_USAGE = 5,
    PERF_COUNTER_MAX = 6
} tpm2_perf_counter_t;

/* Performance statistics */
typedef struct {
    uint64_t total_operations;
    uint64_t successful_operations;
    uint64_t failed_operations;
    double average_latency_us;
    double peak_latency_us;
    uint64_t total_bytes_processed;
    double throughput_mbps;
} tpm2_perf_stats_t;

/**
 * Initialize performance profiling
 */
tpm2_rc_t tpm2_profiling_init(void);

/**
 * Start performance measurement
 */
tpm2_rc_t tpm2_profiling_start(
    tpm2_perf_counter_t counter,
    uint64_t *measurement_id_out
);

/**
 * End performance measurement
 */
tpm2_rc_t tpm2_profiling_end(
    uint64_t measurement_id
);

/**
 * Get performance statistics
 */
tpm2_rc_t tpm2_profiling_get_stats(
    tpm2_perf_counter_t counter,
    tpm2_perf_stats_t *stats_out
);

/**
 * Reset performance counters
 */
tpm2_rc_t tpm2_profiling_reset(
    tpm2_perf_counter_t counter
);

/**
 * Export performance report
 */
tpm2_rc_t tpm2_profiling_export_report(
    const char *filename,
    bool include_raw_data
);

/**
 * Cleanup performance profiling
 */
void tpm2_profiling_cleanup(void);

/* =============================================================================
 * LIBRARY INITIALIZATION AND CONFIGURATION
 * =============================================================================
 */

/* Global library configuration */
typedef struct {
    tpm2_security_level_t security_level;
    tpm2_acceleration_flags_t acceleration_flags;
    bool enable_profiling;
    bool enable_fault_detection;
    uint32_t max_sessions;
    uint32_t memory_pool_size_mb;
    const char *log_file_path;
    bool enable_debug_mode;
} tpm2_library_config_t;

/**
 * Initialize the entire TPM2 compatibility acceleration library
 */
tpm2_rc_t tpm2_library_init(
    const tpm2_library_config_t *config
);

/**
 * Get library version information
 */
const char* tpm2_library_get_version(void);

/**
 * Get library build information
 */
const char* tpm2_library_get_build_info(void);

/**
 * Get enabled acceleration features
 */
tpm2_acceleration_flags_t tpm2_library_get_acceleration_features(void);

/**
 * Cleanup entire library and free resources
 */
void tpm2_library_cleanup(void);

/* =============================================================================
 * ERROR HANDLING AND DEBUGGING
 * =============================================================================
 */

/**
 * Convert return code to human-readable string
 */
const char* tpm2_rc_to_string(tpm2_rc_t rc);

/**
 * Get last error details
 */
tpm2_rc_t tpm2_get_last_error(
    char *error_message,
    size_t message_size,
    uint32_t *error_code_out
);

/**
 * Enable debug logging
 */
tpm2_rc_t tpm2_debug_enable(
    const char *log_file_path,
    bool verbose_mode
);

/**
 * Disable debug logging
 */
void tpm2_debug_disable(void);

#ifdef __cplusplus
}
#endif

#endif /* TPM2_COMPAT_ACCELERATED_H */