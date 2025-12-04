/**
 * @file dsmil_device47_crypto_runtime.c
 * @brief Device 47 (AI/ML) Crypto Integration with Device 255
 * 
 * Integrates Device 47 model encryption/decryption and signing
 * with Device 255 Master Crypto Controller.
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_device255_crypto.h"
#include "dsmil_layer7_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEVICE47_ID 47
#define DEVICE47_LAYER 7
#define AES256_KEY_SIZE 32
#define AES256_IV_SIZE 16

/**
 * @brief Encrypt LLM model using Device 255 (AES-256-GCM)
 * 
 * @param model_data Model data
 * @param model_size Model size
 * @param key Encryption key (32 bytes)
 * @param encrypted_output Output buffer
 * @param encrypted_size Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device47_encrypt_model(const void *model_data, size_t model_size,
                                  const void *key,
                                  void *encrypted_output, size_t *encrypted_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7 (EXTENDED)
    if (dsmil_device255_init(DEVICE47_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Use hardware acceleration for performance
    if (dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_HARDWARE) != 0) {
        fprintf(stderr, "WARNING: Hardware engine unavailable, falling back to TPM\n");
        dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_TPM);
    }
    
    // Generate IV
    uint8_t iv[AES256_IV_SIZE];
    dsmil_crypto_engine_t rng_source;
    
    if (dsmil_device255_rng(&device255_ctx, iv, sizeof(iv), &rng_source) != 0) {
        fprintf(stderr, "ERROR: Failed to generate IV\n");
        return -1;
    }
    
    // Encrypt model (AES-256-GCM)
    if (dsmil_device255_encrypt(&device255_ctx, TPM_ALG_AES,
                                key, AES256_KEY_SIZE,
                                iv, sizeof(iv),
                                model_data, model_size,
                                encrypted_output, encrypted_size) != 0) {
        fprintf(stderr, "ERROR: Failed to encrypt model\n");
        return -1;
    }
    
    return 0;
}

/**
 * @brief Decrypt LLM model using Device 255
 * 
 * @param encrypted_data Encrypted model data
 * @param encrypted_size Encrypted size
 * @param key Decryption key (32 bytes)
 * @param iv Initialization vector (16 bytes)
 * @param decrypted_output Output buffer
 * @param decrypted_size Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device47_decrypt_model(const void *encrypted_data, size_t encrypted_size,
                                  const void *key, const void *iv,
                                  void *decrypted_output, size_t *decrypted_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7
    if (dsmil_device255_init(DEVICE47_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Use hardware acceleration
    dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_HARDWARE);
    
    // Decrypt model
    if (dsmil_device255_decrypt(&device255_ctx, TPM_ALG_AES,
                                key, AES256_KEY_SIZE,
                                iv, AES256_IV_SIZE,
                                encrypted_data, encrypted_size,
                                decrypted_output, decrypted_size) != 0) {
        fprintf(stderr, "ERROR: Failed to decrypt model\n");
        return -1;
    }
    
    return 0;
}

/**
 * @brief Sign model using Device 255 (ML-DSA-87 CNSA 2.0)
 * 
 * @param model_data Model data
 * @param model_size Model size
 * @param private_key Private key for signing
 * @param key_size Key size
 * @param signature Output signature buffer
 * @param signature_size Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_device47_sign_model(const void *model_data, size_t model_size,
                              const void *private_key, size_t key_size,
                              void *signature, size_t *signature_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7
    if (dsmil_device255_init(DEVICE47_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Verify PQC availability
    if (!dsmil_device255_pqc_available(&device255_ctx, TPM_ALG_ML_DSA_87)) {
        fprintf(stderr, "ERROR: ML-DSA-87 not available\n");
        return -1;
    }
    
    // Hash model first (SHA-384)
    uint8_t hash[48];
    size_t hash_len = sizeof(hash);
    
    if (dsmil_device255_hash(&device255_ctx, TPM_ALG_SHA384,
                             model_data, model_size,
                             hash, &hash_len) != 0) {
        fprintf(stderr, "ERROR: Failed to hash model\n");
        return -1;
    }
    
    // Sign hash using ML-DSA-87
    if (dsmil_device255_sign(&device255_ctx, TPM_ALG_ML_DSA_87,
                             private_key, key_size,
                             hash, hash_len,
                             signature, signature_size) != 0) {
        fprintf(stderr, "ERROR: Failed to sign model\n");
        return -1;
    }
    
    return 0;
}

/**
 * @brief Verify model signature using Device 255
 * 
 * @param model_data Model data
 * @param model_size Model size
 * @param public_key Public key for verification
 * @param key_size Key size
 * @param signature Signature to verify
 * @param signature_size Signature size
 * @return 0 if valid, negative if invalid
 */
int dsmil_device47_verify_model_signature(const void *model_data, size_t model_size,
                                          const void *public_key, size_t key_size,
                                          const void *signature, size_t signature_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7
    if (dsmil_device255_init(DEVICE47_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Verify PQC availability
    if (!dsmil_device255_pqc_available(&device255_ctx, TPM_ALG_ML_DSA_87)) {
        fprintf(stderr, "ERROR: ML-DSA-87 not available\n");
        return -1;
    }
    
    // Hash model
    uint8_t hash[48];
    size_t hash_len = sizeof(hash);
    
    if (dsmil_device255_hash(&device255_ctx, TPM_ALG_SHA384,
                             model_data, model_size,
                             hash, &hash_len) != 0) {
        fprintf(stderr, "ERROR: Failed to hash model\n");
        return -1;
    }
    
    // Verify signature
    if (dsmil_device255_verify(&device255_ctx, TPM_ALG_ML_DSA_87,
                               public_key, key_size,
                               hash, hash_len,
                               signature, signature_size) != 0) {
        fprintf(stderr, "ERROR: Model signature verification failed\n");
        return -1;
    }
    
    return 0;
}
