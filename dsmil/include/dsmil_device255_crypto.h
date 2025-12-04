/**
 * @file dsmil_device255_crypto.h
 * @brief Device 255 (Master Crypto Controller) Runtime API
 * 
 * Provides runtime interface to Device 255 unified crypto subsystem:
 * - 88 algorithms (TPM + Hardware + Software)
 * - 3 engines (TPM 2.0, Hardware acceleration, Software fallback)
 * - Layer-aware crypto operations
 * - TPM-protected operations
 * - PQC algorithm support
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_DEVICE255_CRYPTO_H
#define DSMIL_DEVICE255_CRYPTO_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_DEVICE255 Device 255 Master Crypto Controller
 * @{
 */

/**
 * @brief Device 255 device ID
 */
#define DSMIL_DEVICE255_ID 255
#define DSMIL_DEVICE255_TOKEN_BASE 0x8600
#define DSMIL_DEVICE255_TOKEN_MAX 0x86FF

/**
 * @brief Crypto engine types
 */
typedef enum {
    DSMIL_CRYPTO_ENGINE_TPM = 0,      // TPM 2.0 (default)
    DSMIL_CRYPTO_ENGINE_HARDWARE = 1, // Intel AES-NI/SHA-NI/AVX-512
    DSMIL_CRYPTO_ENGINE_SOFTWARE = 2  // Kernel crypto API fallback
} dsmil_crypto_engine_t;

/**
 * @brief Algorithm categories (bits 0-9)
 */
#define DSMIL_CRYPTO_CAP_HASH           (1 << 0)  // 10 algorithms
#define DSMIL_CRYPTO_CAP_SYMMETRIC      (1 << 1)  // 22 algorithms
#define DSMIL_CRYPTO_CAP_ASYMMETRIC     (1 << 2)  // 5 algorithms
#define DSMIL_CRYPTO_CAP_ECC            (1 << 3)  // 12 algorithms
#define DSMIL_CRYPTO_CAP_KDF            (1 << 4)  // 11 algorithms
#define DSMIL_CRYPTO_CAP_HMAC           (1 << 5)  // 5 algorithms
#define DSMIL_CRYPTO_CAP_SIGNATURES     (1 << 6)  // 8 algorithms
#define DSMIL_CRYPTO_CAP_KEY_AGREEMENT  (1 << 7)  // 3 algorithms
#define DSMIL_CRYPTO_CAP_MGF            (1 << 8)  // 4 algorithms
#define DSMIL_CRYPTO_CAP_POST_QUANTUM   (1 << 9)  // 8 algorithms

/**
 * @brief Hardware acceleration flags (bits 10-15)
 */
#define DSMIL_CRYPTO_CAP_AES_NI         (1 << 10)
#define DSMIL_CRYPTO_CAP_SHA_NI        (1 << 11)
#define DSMIL_CRYPTO_CAP_AVX512        (1 << 12)
#define DSMIL_CRYPTO_CAP_TPM           (1 << 13)
#define DSMIL_CRYPTO_CAP_SECURE_BOOT   (1 << 14)
#define DSMIL_CRYPTO_CAP_KEY_STORAGE   (1 << 15)

/**
 * @brief Convenience masks
 */
#define DSMIL_CRYPTO_CAP_ALL_ALGOS  0x03FF  /* Bits 0-9 */
#define DSMIL_CRYPTO_CAP_ALL_HW     0xFC00  /* Bits 10-15 */
#define DSMIL_CRYPTO_CAP_ALL        0xFFFF  /* All capabilities */

/**
 * @brief Algorithm IDs (simplified - actual would use TPM_ALG_*)
 */
#define TPM_ALG_SHA256    0x000B
#define TPM_ALG_SHA384    0x000C
#define TPM_ALG_SHA512    0x000D
#define TPM_ALG_AES       0x0006
#define TPM_ALG_RSA       0x0001
#define TPM_ALG_ECDSA     0x0018
#define TPM_ALG_ML_KEM_1024 0x0026  // Placeholder
#define TPM_ALG_ML_DSA_87   0x0027  // Placeholder

/**
 * @brief Device 255 capability registers
 */
typedef struct {
    uint16_t available;    // Detected at probe time
    uint16_t enabled;      // Runtime enable/disable
    uint16_t locked;       // TPM-protected lock
    uint32_t active_engine;
    uint32_t algorithm_count;  // 88 total
    uint64_t total_operations;
    bool tpm_available;
    bool secure_boot_verified;
} dsmil_device255_caps_t;

/**
 * @brief Device 255 context
 */
typedef struct {
    uint32_t device_id;           // 255
    uint8_t layer;                // Current layer context
    dsmil_crypto_engine_t engine;
    dsmil_device255_caps_t caps;
    uint32_t operation_count;
    uint64_t bytes_processed;
} dsmil_device255_ctx_t;

/**
 * @brief Initialize Device 255 crypto subsystem
 * 
 * @param layer Layer context (0-9)
 * @param ctx Output context
 * @return 0 on success, negative on error
 */
int dsmil_device255_init(uint8_t layer, dsmil_device255_ctx_t *ctx);

/**
 * @brief Get capability registers
 * 
 * @param ctx Device 255 context
 * @param caps Output capabilities
 * @return 0 on success, negative on error
 */
int dsmil_device255_get_caps(const dsmil_device255_ctx_t *ctx,
                              dsmil_device255_caps_t *caps);

/**
 * @brief Set active crypto engine
 * 
 * @param ctx Device 255 context
 * @param engine Engine type (TPM/Hardware/Software)
 * @return 0 on success, negative on error
 */
int dsmil_device255_set_engine(dsmil_device255_ctx_t *ctx,
                                dsmil_crypto_engine_t engine);

/**
 * @brief Hash operation
 * 
 * @param ctx Device 255 context
 * @param algorithm Algorithm ID (TPM_ALG_SHA256, etc.)
 * @param input Input data
 * @param input_len Input length
 * @param output Output buffer
 * @param output_len Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device255_hash(const dsmil_device255_ctx_t *ctx,
                         uint16_t algorithm,
                         const void *input, size_t input_len,
                         void *output, size_t *output_len);

/**
 * @brief Encrypt operation
 * 
 * @param ctx Device 255 context
 * @param algorithm Algorithm ID
 * @param key Encryption key
 * @param key_len Key length
 * @param iv Initialization vector
 * @param iv_len IV length
 * @param plaintext Input plaintext
 * @param plaintext_len Plaintext length
 * @param ciphertext Output ciphertext buffer
 * @param ciphertext_len Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device255_encrypt(const dsmil_device255_ctx_t *ctx,
                            uint16_t algorithm,
                            const void *key, size_t key_len,
                            const void *iv, size_t iv_len,
                            const void *plaintext, size_t plaintext_len,
                            void *ciphertext, size_t *ciphertext_len);

/**
 * @brief Decrypt operation
 * 
 * @param ctx Device 255 context
 * @param algorithm Algorithm ID
 * @param key Decryption key
 * @param key_len Key length
 * @param iv Initialization vector
 * @param iv_len IV length
 * @param ciphertext Input ciphertext
 * @param ciphertext_len Ciphertext length
 * @param plaintext Output plaintext buffer
 * @param plaintext_len Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device255_decrypt(const dsmil_device255_ctx_t *ctx,
                            uint16_t algorithm,
                            const void *key, size_t key_len,
                            const void *iv, size_t iv_len,
                            const void *ciphertext, size_t ciphertext_len,
                            void *plaintext, size_t *plaintext_len);

/**
 * @brief Sign operation
 * 
 * @param ctx Device 255 context
 * @param algorithm Algorithm ID (RSA-SSA, ECDSA, ML-DSA-87, etc.)
 * @param private_key Private key
 * @param key_len Key length
 * @param message Message to sign
 * @param message_len Message length
 * @param signature Output signature buffer
 * @param signature_len Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device255_sign(const dsmil_device255_ctx_t *ctx,
                         uint16_t algorithm,
                         const void *private_key, size_t key_len,
                         const void *message, size_t message_len,
                         void *signature, size_t *signature_len);

/**
 * @brief Verify signature
 * 
 * @param ctx Device 255 context
 * @param algorithm Algorithm ID
 * @param public_key Public key
 * @param key_len Key length
 * @param message Original message
 * @param message_len Message length
 * @param signature Signature to verify
 * @param signature_len Signature length
 * @return 0 if valid, negative if invalid
 */
int dsmil_device255_verify(const dsmil_device255_ctx_t *ctx,
                           uint16_t algorithm,
                           const void *public_key, size_t key_len,
                           const void *message, size_t message_len,
                           const void *signature, size_t signature_len);

/**
 * @brief Get random bytes
 * 
 * @param ctx Device 255 context
 * @param output Output buffer
 * @param len Requested length
 * @param source Output source engine
 * @return 0 on success, negative on error
 */
int dsmil_device255_rng(const dsmil_device255_ctx_t *ctx,
                        void *output, size_t len,
                        dsmil_crypto_engine_t *source);

/**
 * @brief Secure data wipe (TPM-protected)
 * 
 * @param ctx Device 255 context
 * @param target Wipe target bitmask
 * @param confirmation Confirmation code (0xDEADBEEF)
 * @param session_token TPM session token
 * @return 0 on success, negative on error
 */
int dsmil_device255_data_wipe(dsmil_device255_ctx_t *ctx,
                              uint32_t target,
                              uint32_t confirmation,
                              uint32_t session_token);

/**
 * @brief Enable/disable capability
 * 
 * @param ctx Device 255 context
 * @param capability Capability flag
 * @param enable true to enable, false to disable
 * @return 0 on success, negative on error
 */
int dsmil_device255_cap_control(dsmil_device255_ctx_t *ctx,
                                uint16_t capability,
                                bool enable);

/**
 * @brief Lock capability (TPM-protected)
 * 
 * @param ctx Device 255 context
 * @param capability Capability flag
 * @param session_token TPM session token
 * @return 0 on success, negative on error
 */
int dsmil_device255_cap_lock(dsmil_device255_ctx_t *ctx,
                             uint16_t capability,
                             uint32_t session_token);

/**
 * @brief Check if PQC algorithm is available
 * 
 * @param ctx Device 255 context
 * @param pqc_algorithm PQC algorithm ID (ML-KEM-1024, ML-DSA-87, etc.)
 * @return true if available, false otherwise
 */
bool dsmil_device255_pqc_available(const dsmil_device255_ctx_t *ctx,
                                   uint16_t pqc_algorithm);

/**
 * @brief Get operation statistics
 * 
 * @param ctx Device 255 context
 * @param total_ops Output total operations
 * @param bytes_processed Output bytes processed
 * @param engine_stats Output stats per engine
 * @return 0 on success, negative on error
 */
int dsmil_device255_get_stats(const dsmil_device255_ctx_t *ctx,
                              uint64_t *total_ops,
                              uint64_t *bytes_processed,
                              uint64_t engine_stats[3]);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_DEVICE255_CRYPTO_H */
