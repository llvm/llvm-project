# Device 255 (Master Crypto Controller) Enhancement Plan

**Version**: 1.0.0  
**Date**: 2025-01-15  
**Status**: Enhancement Recommendations  
**Device**: 255 (0xFF) - Master Crypto Controller  
**Reference**: Comprehensive Plan for Kitty + AI / Kernel Dev

---

## Executive Summary

Device 255 (Master Crypto Controller) is the unified cryptographic subsystem providing centralized management of all cryptographic operations across the DSMIL 104-device architecture. This enhancement plan integrates Device 255 with:

1. **Layer 3 Device 15 (CRYPTO)** - Wycheproof integration and crypto assurance
2. **Layer 8 (ENHANCED_SEC)** - Security AI and PQC enforcement
3. **Device 47 (Advanced AI/ML)** - Model encryption, signing, and secure storage
4. **Device 46 (Quantum)** - Post-quantum cryptography (PQC) support
5. **MLOps Pipeline** - Model provenance, signing, and verification
6. **Cross-Layer Intelligence Flows** - Secure communication and encryption
7. **Hardware Integration Layer (HIL)** - NPU/GPU/CPU crypto acceleration

---

## 1. Architecture Integration

### 1.1 Device 255 in DSMIL Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Device 255: Master Crypto Controller               │
│              Token Range: 0x8600-0x86FF                         │
│              Algorithms: 88 (TPM + Hardware + Software)         │
│              Layer Access: All Layers (0-9)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Layer 3     │    │  Layer 7      │    │  Layer 8      │
│  Device 15   │    │  Device 47     │    │  Device 51-58 │
│  (CRYPTO)    │    │  (AI/ML)       │    │  (SECURITY)   │
│              │    │                │    │               │
│ Wycheproof   │    │ Model Signing │    │ PQC Enforce   │
│ Crypto Tests │    │ Secure Storage│    │ Zero Trust    │
│              │    │ Key Management│    │ Threat Intel  │
└──────────────┘    └───────────────┘    └───────────────┘
```

### 1.2 Device 255 Capability Mapping to DSMIL Layers

| Layer | Primary Use Case | Device 255 Capabilities |
|-------|------------------|------------------------|
| Layer 3 (SECRET) | Device 15 Wycheproof crypto operations | All symmetric/hash algorithms, TPM attestation |
| Layer 4 (TOP_SECRET) | Mission-critical encryption | RSA-4096, ECC P-521, TPM-protected keys |
| Layer 5 (COSMIC) | Predictive analytics encryption | AES-256, SHA-384, HMAC |
| Layer 6 (ATOMAL) | Nuclear command encryption | RSA-8192, ECC P-521, TPM PCR attestation |
| Layer 7 (EXTENDED) | Device 47 model encryption/signing | AES-256, RSA-4096, PQC (ML-KEM, ML-DSA) |
| Layer 8 (ENHANCED_SEC) | Security AI, PQC enforcement | All PQC algorithms, TPM attestation |
| Layer 9 (EXECUTIVE) | Strategic command encryption | RSA-8192, ECC P-521, TPM-protected operations |

---

## 2. DSMIL Folder Enhancements

### 2.1 Device 255 Runtime API

#### New Header: `dsmil/include/dsmil_device255_crypto.h`

```c
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

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_DEVICE255_CRYPTO_H */
```

**Implementation File**: `dsmil/lib/Runtime/dsmil_device255_crypto_runtime.c`

---

### 2.2 Device 15 (CRYPTO) Integration with Device 255

#### Enhancement: Wycheproof Runtime Uses Device 255

**New File**: `dsmil/lib/Runtime/dsmil_device15_wycheproof_runtime.c`

```c
/**
 * @file dsmil_device15_wycheproof_runtime.c
 * @brief Device 15 (CRYPTO) Wycheproof Runtime with Device 255 Integration
 * 
 * Device 15 uses Device 255 for all cryptographic operations:
 * - Hash operations for test vectors
 * - Encryption/decryption for crypto library testing
 * - Signature verification
 * - RNG for test vector generation
 */

#include "dsmil_device255_crypto.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEVICE15_ID 15
#define DEVICE15_LAYER 3

/**
 * @brief Device 15 Wycheproof context
 */
typedef struct {
    dsmil_device255_ctx_t crypto_ctx;  // Uses Device 255
    uint32_t device_id;                 // 15
    uint8_t layer;                      // 3
    uint64_t memory_budget_bytes;       // 6 GB Layer 3 max
    uint64_t memory_used_bytes;
    uint64_t test_vectors_processed;
    uint64_t crypto_operations;
} dsmil_device15_ctx_t;

static dsmil_device15_ctx_t g_device15_ctx = {0};

/**
 * @brief Initialize Device 15 with Device 255 crypto
 */
int dsmil_device15_wycheproof_init(void) {
    if (g_device15_ctx.crypto_ctx.device_id != 0) {
        return 0;  // Already initialized
    }
    
    g_device15_ctx.device_id = DEVICE15_ID;
    g_device15_ctx.layer = DEVICE15_LAYER;
    g_device15_ctx.memory_budget_bytes = 6ULL * 1024 * 1024 * 1024;  // 6 GB
    
    // Initialize Device 255 crypto subsystem
    if (dsmil_device255_init(DEVICE15_LAYER, &g_device15_ctx.crypto_ctx) != 0) {
        return -1;
    }
    
    // Use TPM engine for Device 15 (attestation required)
    dsmil_device255_set_engine(&g_device15_ctx.crypto_ctx,
                                DSMIL_CRYPTO_ENGINE_TPM);
    
    return 0;
}

/**
 * @brief Hash test vector using Device 255
 */
int dsmil_device15_hash_vector(const void *data, size_t data_len,
                                uint16_t algorithm,
                                void *hash_output, size_t *hash_len) {
    if (g_device15_ctx.crypto_ctx.device_id == 0) {
        dsmil_device15_wycheproof_init();
    }
    
    g_device15_ctx.crypto_operations++;
    
    return dsmil_device255_hash(&g_device15_ctx.crypto_ctx,
                                 algorithm,
                                 data, data_len,
                                 hash_output, hash_len);
}

/**
 * @brief Encrypt test vector using Device 255
 */
int dsmil_device15_encrypt_vector(const void *key, size_t key_len,
                                   const void *iv, size_t iv_len,
                                   const void *plaintext, size_t plaintext_len,
                                   uint16_t algorithm,
                                   void *ciphertext, size_t *ciphertext_len) {
    if (g_device15_ctx.crypto_ctx.device_id == 0) {
        dsmil_device15_wycheproof_init();
    }
    
    g_device15_ctx.crypto_operations++;
    
    return dsmil_device255_encrypt(&g_device15_ctx.crypto_ctx,
                                    algorithm,
                                    key, key_len,
                                    iv, iv_len,
                                    plaintext, plaintext_len,
                                    ciphertext, ciphertext_len);
}

/**
 * @brief Get random bytes for test vector generation
 */
int dsmil_device15_get_random(void *output, size_t len) {
    if (g_device15_ctx.crypto_ctx.device_id == 0) {
        dsmil_device15_wycheproof_init();
    }
    
    dsmil_crypto_engine_t source;
    return dsmil_device255_rng(&g_device15_ctx.crypto_ctx,
                               output, len,
                               &source);
}
```

---

### 2.3 Device 47 (AI/ML) Integration with Device 255

#### Enhancement: Model Encryption and Signing

**New File**: `dsmil/lib/Runtime/dsmil_device47_crypto_runtime.c`

```c
/**
 * @file dsmil_device47_crypto_runtime.c
 * @brief Device 47 (AI/ML) Crypto Integration with Device 255
 * 
 * Device 47 uses Device 255 for:
 * - Model encryption/decryption (secure storage)
 * - Model signing (provenance, CNSA 2.0)
 * - Key management for encrypted models
 * - INT8 quantization key protection
 */

#include "dsmil_device255_crypto.h"
#include "dsmil_layer7_llm.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Encrypt LLM model for secure storage
 * 
 * Uses Device 255 AES-256-GCM encryption
 */
int dsmil_device47_encrypt_model(const dsmil_device47_llm_ctx_t *llm_ctx,
                                  const void *model_data, size_t model_size,
                                  void *encrypted_model, size_t *encrypted_size,
                                  void *key, size_t *key_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Use hardware engine for performance (Layer 7)
    dsmil_device255_set_engine(&crypto_ctx, DSMIL_CRYPTO_ENGINE_HARDWARE);
    
    // Generate encryption key
    uint8_t iv[12];  // GCM nonce
    dsmil_device255_rng(&crypto_ctx, key, 32, NULL);  // AES-256 key
    dsmil_device255_rng(&crypto_ctx, iv, sizeof(iv), NULL);
    
    // Encrypt model
    return dsmil_device255_encrypt(&crypto_ctx,
                                   TPM_ALG_AES,  // AES-256-GCM
                                   key, 32,
                                   iv, sizeof(iv),
                                   model_data, model_size,
                                   encrypted_model, encrypted_size);
}

/**
 * @brief Sign LLM model with ML-DSA-87 (CNSA 2.0)
 * 
 * Uses Device 255 PQC signature support
 */
int dsmil_device47_sign_model(const dsmil_device47_llm_ctx_t *llm_ctx,
                               const void *model_data, size_t model_size,
                               const void *private_key, size_t key_len,
                               void *signature, size_t *signature_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Verify PQC support
    if (!dsmil_device255_pqc_available(&crypto_ctx, TPM_ALG_ML_DSA_87)) {
        return -1;  // PQC not available
    }
    
    // Use TPM engine for signing (attestation)
    dsmil_device255_set_engine(&crypto_ctx, DSMIL_CRYPTO_ENGINE_TPM);
    
    // Sign model with ML-DSA-87
    return dsmil_device255_sign(&crypto_ctx,
                                TPM_ALG_ML_DSA_87,
                                private_key, key_len,
                                model_data, model_size,
                                signature, signature_len);
}

/**
 * @brief Verify model signature (CNSA 2.0 provenance)
 */
int dsmil_device47_verify_model_signature(const void *model_data, size_t model_size,
                                           const void *public_key, size_t key_len,
                                           const void *signature, size_t signature_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    return dsmil_device255_verify(&crypto_ctx,
                                   TPM_ALG_ML_DSA_87,
                                   public_key, key_len,
                                   model_data, model_size,
                                   signature, signature_len);
}
```

---

### 2.4 Device 46 (Quantum) Integration with Device 255

#### Enhancement: PQC Algorithm Support

**New File**: `dsmil/lib/Runtime/dsmil_device46_pqc_runtime.c`

```c
/**
 * @file dsmil_device46_pqc_runtime.c
 * @brief Device 46 (Quantum) PQC Integration with Device 255
 * 
 * Device 46 uses Device 255 for:
 * - PQC algorithm support (ML-KEM-1024, ML-DSA-87)
 * - Quantum-safe key generation
 * - PQC test vector generation for Wycheproof
 */

#include "dsmil_device255_crypto.h"
#include "dsmil_quantum_runtime.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Generate PQC key pair using Device 255
 */
int dsmil_device46_generate_pqc_keys(dsmil_device46_quantum_ctx_t *quantum_ctx,
                                     uint16_t pqc_algorithm,
                                     void *public_key, size_t *public_key_len,
                                     void *private_key, size_t *private_key_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Verify PQC algorithm is available
    if (!dsmil_device255_pqc_available(&crypto_ctx, pqc_algorithm)) {
        return -1;
    }
    
    // Use TPM engine for key generation (attestation)
    dsmil_device255_set_engine(&crypto_ctx, DSMIL_CRYPTO_ENGINE_TPM);
    
    // Generate random seed for PQC key generation
    uint8_t seed[64];
    dsmil_device255_rng(&crypto_ctx, seed, sizeof(seed), NULL);
    
    // Generate PQC key pair (implementation depends on algorithm)
    // ML-KEM-1024: public_key_len = 1568, private_key_len = 3168
    // ML-DSA-87: public_key_len = 1952, private_key_len = 4000
    
    return 0;  // Placeholder - actual implementation needed
}

/**
 * @brief Generate PQC test vector for Device 15 Wycheproof
 */
int dsmil_device46_generate_pqc_test_vector(uint16_t pqc_algorithm,
                                             void *test_vector,
                                             size_t *test_vector_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Generate PQC key material
    uint8_t public_key[2048];
    uint8_t private_key[4096];
    size_t pk_len = sizeof(public_key);
    size_t sk_len = sizeof(private_key);
    
    if (dsmil_device46_generate_pqc_keys(NULL, pqc_algorithm,
                                         public_key, &pk_len,
                                         private_key, &sk_len) != 0) {
        return -1;
    }
    
    // Format as Wycheproof test vector
    // (conforms to crypto_test_vector_pqc.schema.yaml)
    
    return 0;
}
```

---

### 2.5 Layer 8 (ENHANCED_SEC) Integration with Device 255

#### Enhancement: Security AI Crypto Operations

**New File**: `dsmil/lib/Runtime/dsmil_layer8_security_crypto_runtime.c`

```c
/**
 * @file dsmil_layer8_security_crypto_runtime.c
 * @brief Layer 8 (ENHANCED_SEC) Crypto Integration with Device 255
 * 
 * Layer 8 security devices use Device 255 for:
 * - PQC enforcement (ML-KEM-1024, ML-DSA-87)
 * - Zero-trust key management
 * - Threat intelligence encryption
 * - Security AI model protection
 */

#include "dsmil_device255_crypto.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Enforce PQC-only mode (disable classical crypto)
 */
int dsmil_layer8_enforce_pqc_only(void) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(8, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Disable classical algorithms (keep only PQC)
    dsmil_device255_cap_control(&crypto_ctx,
                                DSMIL_CRYPTO_CAP_ASYMMETRIC,  // Disable RSA
                                false);
    dsmil_device255_cap_control(&crypto_ctx,
                                DSMIL_CRYPTO_CAP_ECC,  // Disable classical ECC
                                false);
    
    // Ensure PQC is enabled
    dsmil_device255_cap_control(&crypto_ctx,
                                DSMIL_CRYPTO_CAP_POST_QUANTUM,
                                true);
    
    return 0;
}

/**
 * @brief Verify PQC algorithm usage (Layer 8 requirement)
 */
bool dsmil_layer8_verify_pqc_usage(uint16_t algorithm) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(8, &crypto_ctx) != 0) {
        return false;
    }
    
    // Check if algorithm is PQC
    return dsmil_device255_pqc_available(&crypto_ctx, algorithm);
}
```

---

### 2.6 MLOps Pipeline Integration with Device 255

#### Enhancement: Model Provenance and Signing

**New File**: `dsmil/lib/Runtime/dsmil_mlops_crypto_runtime.c`

```c
/**
 * @file dsmil_mlops_crypto_runtime.c
 * @brief MLOps Pipeline Crypto Integration with Device 255
 * 
 * MLOps pipeline uses Device 255 for:
 * - Model signing (CNSA 2.0: ML-DSA-87)
 * - INT8 quantization key protection
 * - Model encryption for secure storage
 * - Provenance verification
 */

#include "dsmil_device255_crypto.h"
#include "dsmil_mlops_optimization.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Sign optimized model with ML-DSA-87 (CNSA 2.0)
 */
int dsmil_mlops_sign_model(const char *model_path,
                            const void *private_key, size_t key_len,
                            void *signature, size_t *signature_len,
                            void *provenance_metadata, size_t *metadata_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return -1;
    }
    
    // Verify PQC support
    if (!dsmil_device255_pqc_available(&crypto_ctx, TPM_ALG_ML_DSA_87)) {
        return -1;
    }
    
    // Read model file
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        return -1;
    }
    
    // Get model size and hash
    fseek(f, 0, SEEK_END);
    size_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, f);
    fclose(f);
    
    // Hash model (SHA-384 per CNSA 2.0)
    uint8_t model_hash[48];
    size_t hash_len = sizeof(model_hash);
    dsmil_device255_hash(&crypto_ctx,
                         TPM_ALG_SHA384,
                         model_data, model_size,
                         model_hash, &hash_len);
    
    // Sign hash with ML-DSA-87
    int result = dsmil_device255_sign(&crypto_ctx,
                                      TPM_ALG_ML_DSA_87,
                                      private_key, key_len,
                                      model_hash, hash_len,
                                      signature, signature_len);
    
    free(model_data);
    return result;
}

/**
 * @brief Verify model signature (MLOps gate)
 */
bool dsmil_mlops_verify_model_signature(const char *model_path,
                                        const void *public_key, size_t key_len,
                                        const void *signature, size_t signature_len) {
    dsmil_device255_ctx_t crypto_ctx;
    if (dsmil_device255_init(7, &crypto_ctx) != 0) {
        return false;
    }
    
    // Read and hash model
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        return false;
    }
    
    fseek(f, 0, SEEK_END);
    size_t model_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, f);
    fclose(f);
    
    uint8_t model_hash[48];
    size_t hash_len = sizeof(model_hash);
    dsmil_device255_hash(&crypto_ctx,
                         TPM_ALG_SHA384,
                         model_data, model_size,
                         model_hash, &hash_len);
    
    // Verify signature
    int result = dsmil_device255_verify(&crypto_ctx,
                                        TPM_ALG_ML_DSA_87,
                                        public_key, key_len,
                                        model_hash, hash_len,
                                        signature, signature_len);
    
    free(model_data);
    return (result == 0);
}
```

---

## 3. DSMIL-Wycheproof-Bundle Enhancements

### 3.1 Device 255 Integration Configuration

**New File**: `dsmil-wycheproof-bundle/config/device255_integration.yaml`

```yaml
# Device 255 (Master Crypto Controller) Integration Configuration
# For Device 15 (CRYPTO) Wycheproof Operations

version: 1
device255_id: 255
token_base: 0x8600

integration:
  device15_wycheproof:
    enabled: true
    engine_preference: "TPM"  # TPM, Hardware, Software
    use_tpm_attestation: true
    pcr_index: 17  # Device 15 crypto operations
    
  device47_model_signing:
    enabled: true
    algorithm: "ML-DSA-87"  # CNSA 2.0
    key_storage: "TPM"  # TPM-protected keys
    
  device46_pqc_vectors:
    enabled: true
    algorithms:
      - "ML-KEM-1024"
      - "ML-DSA-87"
      - "Falcon-1024"
    engine: "TPM"  # PQC requires TPM
    
  layer8_security:
    enabled: true
    enforce_pqc_only: true
    disable_classical: true

capabilities:
  # Algorithm categories (bits 0-9)
  hash: true              # Bit 0
  symmetric: true          # Bit 1
  asymmetric: false       # Bit 2 (disabled for PQC-only)
  ecc: false              # Bit 3 (disabled for PQC-only)
  kdf: true               # Bit 4
  hmac: true              # Bit 5
  signatures: true        # Bit 6
  key_agreement: true     # Bit 7
  mgf: true               # Bit 8
  post_quantum: true      # Bit 9 (REQUIRED)
  
  # Hardware acceleration (bits 10-15)
  aes_ni: true            # Bit 10
  sha_ni: true            # Bit 11
  avx512: true            # Bit 12
  tpm: true               # Bit 13 (REQUIRED)
  secure_boot: true       # Bit 14
  key_storage: true       # Bit 15

engines:
  tpm:
    enabled: true
    default: true
    use_for: [ "attestation", "pqc", "key_generation" ]
    
  hardware:
    enabled: true
    use_for: [ "symmetric", "hash", "performance" ]
    
  software:
    enabled: true
    use_for: [ "fallback", "compatibility" ]

protected_operations:
  data_wipe:
    enabled: true
    requires_tpm_session: true
    confirmation_code: 0xDEADBEEF
    
  capability_lock:
    enabled: true
    requires_tpm_session: true
    
  key_management:
    enabled: true
    requires_tpm_session: true
```

---

### 3.2 Enhanced Wycheproof Schema with Device 255 Metadata

**Enhancement**: Update `crypto_test_result.schema.yaml`

```yaml
# Add Device 255 metadata to crypto_test_result schema

properties:
  # ... existing properties ...
  
  device255_metadata:
    type: object
    properties:
      device_id:
        type: integer
        const: 255
      engine_used:
        type: string
        enum: [ "TPM", "Hardware", "Software" ]
      capabilities_used:
        type: object
        properties:
          hash: { type: boolean }
          symmetric: { type: boolean }
          asymmetric: { type: boolean }
          ecc: { type: boolean }
          pqc: { type: boolean }
      tpm_attestation:
        type: object
        properties:
          pcr_index: { type: integer }
          pcr_value: { type: string }
          session_token: { type: string }
      hardware_acceleration:
        type: object
        properties:
          aes_ni: { type: boolean }
          sha_ni: { type: boolean }
          avx512: { type: boolean }
      performance_metrics:
        type: object
        properties:
          operation_time_ns: { type: integer }
          bytes_processed: { type: integer }
          engine_utilization: { type: number }
```

---

### 3.3 Cross-Device Crypto Intelligence Flow

**New File**: `dsmil-wycheproof-bundle/config/device255_intelligence_flows.yaml`

```yaml
# Device 255 Cross-Device Intelligence Flows

version: 1

intelligence_flows:
  # Device 15 (CRYPTO) → Device 255 → Device 47 (AI/ML)
  - source:
      device_id: 15
      layer: 3
    crypto_device:
      device_id: 255
    target:
      device_id: 47
      layer: 7
    flow:
      - step: "Device 15 performs crypto operation"
      - step: "Device 255 provides crypto engine"
      - step: "Device 255 extends PCR 17 (attestation)"
      - step: "Results encrypted with Device 255"
      - step: "Encrypted results sent to Device 47"
      - step: "Device 47 decrypts and analyzes"
    crypto_operations:
      - operation: "hash"
        algorithm: "SHA-384"
        engine: "TPM"
      - operation: "encrypt"
        algorithm: "AES-256-GCM"
        engine: "Hardware"
      - operation: "sign"
        algorithm: "ML-DSA-87"
        engine: "TPM"
        
  # Device 46 (Quantum) → Device 255 → Device 15 (CRYPTO)
  - source:
      device_id: 46
      layer: 7
    crypto_device:
      device_id: 255
    target:
      device_id: 15
      layer: 3
    flow:
      - step: "Device 46 generates PQC test vector"
      - step: "Device 255 generates PQC keys (ML-KEM-1024)"
      - step: "Device 255 signs vector with ML-DSA-87"
      - step: "Signed vector sent to Device 15"
      - step: "Device 15 validates signature"
      - step: "Device 15 executes Wycheproof test"
    crypto_operations:
      - operation: "key_generation"
        algorithm: "ML-KEM-1024"
        engine: "TPM"
      - operation: "sign"
        algorithm: "ML-DSA-87"
        engine: "TPM"
      - operation: "verify"
        algorithm: "ML-DSA-87"
        engine: "TPM"
        
  # Device 47 (AI/ML) → Device 255 → Device 15 (CRYPTO)
  - source:
      device_id: 47
      layer: 7
    crypto_device:
      device_id: 255
    target:
      device_id: 15
      layer: 3
    flow:
      - step: "Device 47 generates AI test vector"
      - step: "Device 255 encrypts vector (AES-256-GCM)"
      - step: "Device 255 signs with ML-DSA-87"
      - step: "Encrypted+signed vector sent to Device 15"
      - step: "Device 15 verifies signature"
      - step: "Device 15 decrypts and tests"
    crypto_operations:
      - operation: "encrypt"
        algorithm: "AES-256-GCM"
        engine: "Hardware"
      - operation: "sign"
        algorithm: "ML-DSA-87"
        engine: "TPM"
      - operation: "decrypt"
        algorithm: "AES-256-GCM"
        engine: "Hardware"
      - operation: "verify"
        algorithm: "ML-DSA-87"
        engine: "TPM"
```

---

## 4. Implementation Priority

### Phase 1: Core Device 255 Runtime (Weeks 1-4)
1. ✅ `dsmil_device255_crypto.h` API definition
2. ✅ `dsmil_device255_crypto_runtime.c` implementation
3. ✅ IOCTL interface integration
4. ✅ Sysfs interface integration
5. ✅ TPM authentication flow

### Phase 2: Device Integration (Weeks 5-8)
1. ✅ Device 15 (CRYPTO) integration with Device 255
2. ✅ Device 47 (AI/ML) model signing/encryption
3. ✅ Device 46 (Quantum) PQC support
4. ✅ Layer 8 (ENHANCED_SEC) PQC enforcement

### Phase 3: MLOps & Intelligence Flows (Weeks 9-12)
1. ✅ MLOps model provenance signing
2. ✅ Cross-device intelligence flow encryption
3. ✅ Wycheproof Device 255 metadata
4. ✅ Device 255 intelligence flow configuration

### Phase 4: Advanced Features (Weeks 13-16)
1. ✅ Hardware acceleration optimization
2. ✅ Multi-engine load balancing
3. ✅ Performance monitoring
4. ✅ Security audit integration

---

## 5. Testing & Validation

### 5.1 Unit Tests
- Device 255 capability detection
- Engine switching (TPM/Hardware/Software)
- Algorithm availability checks
- TPM authentication flow
- Protected operation authorization

### 5.2 Integration Tests
- Device 15 → Device 255 crypto operations
- Device 47 → Device 255 model signing
- Device 46 → Device 255 PQC key generation
- Layer 8 → Device 255 PQC enforcement
- MLOps → Device 255 provenance signing

### 5.3 Performance Tests
- Engine performance comparison (TPM vs Hardware vs Software)
- Throughput per algorithm
- Latency measurements
- Memory usage per operation
- TPM session overhead

### 5.4 Security Tests
- TPM-protected operation authorization
- Capability locking/unlocking
- Secure data wipe verification
- PCR attestation validation
- PQC algorithm correctness

---

## 6. Documentation Updates

### 6.1 DSMIL Documentation

**New Documents**:
- `dsmil/docs/DEVICE255-MASTER-CRYPTO-CONTROLLER.md` - Complete Device 255 guide
- `dsmil/docs/DEVICE255-DEVICE15-INTEGRATION.md` - Device 15 crypto integration
- `dsmil/docs/DEVICE255-DEVICE47-INTEGRATION.md` - Device 47 model crypto
- `dsmil/docs/DEVICE255-DEVICE46-INTEGRATION.md` - Device 46 PQC support
- `dsmil/docs/DEVICE255-MLOPS-INTEGRATION.md` - MLOps provenance signing
- `dsmil/docs/DEVICE255-LAYER8-INTEGRATION.md` - Layer 8 security crypto

### 6.2 Wycheproof Bundle Documentation

**New Documents**:
- `dsmil-wycheproof-bundle/docs/DEVICE255-INTEGRATION.md` - Device 255 integration guide
- `dsmil-wycheproof-bundle/docs/DEVICE255-INTELLIGENCE-FLOWS.md` - Crypto intelligence flows

---

## 7. Summary

These enhancements integrate Device 255 (Master Crypto Controller) with the comprehensive AI system integration plan by:

1. **Unified Crypto API** - Single interface for all 88 algorithms across all layers
2. **Device 15 Integration** - Wycheproof uses Device 255 for all crypto operations
3. **Device 47 Integration** - Model encryption, signing, and secure storage
4. **Device 46 Integration** - PQC algorithm support for quantum-safe crypto
5. **Layer 8 Integration** - PQC enforcement and zero-trust key management
6. **MLOps Integration** - Model provenance signing (CNSA 2.0: ML-DSA-87)
7. **Cross-Layer Intelligence** - Secure encryption for intelligence flows
8. **Hardware Acceleration** - NPU/GPU/CPU crypto optimization via HIL

All enhancements maintain compatibility with existing Device 255 kernel driver while adding high-level runtime APIs for DSMIL application integration.

---

**End of Device 255 Enhancement Plan**
