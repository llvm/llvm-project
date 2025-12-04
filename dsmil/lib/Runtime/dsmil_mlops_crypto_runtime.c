/**
 * @file dsmil_mlops_crypto_runtime.c
 * @brief MLOps Pipeline Crypto Integration with Device 255
 * 
 * Uses Device 255 for model provenance signing (ML-DSA-87 CNSA 2.0)
 * and verification in the MLOps pipeline.
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_device255_crypto.h"
#include "dsmil_mlops_optimization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MLOPS_LAYER 7  // MLOps operates at Layer 7

/**
 * @brief Sign model provenance using Device 255 (ML-DSA-87 CNSA 2.0)
 * 
 * @param model_path Path to model file
 * @param provenance_metadata Provenance metadata
 * @param metadata_size Metadata size
 * @param private_key Private key for signing
 * @param key_size Key size
 * @param signature Output signature buffer
 * @param signature_size Output buffer size / actual length
 * @return 0 on success, negative on error
 */
int dsmil_mlops_sign_provenance(const char *model_path,
                                 const void *provenance_metadata, size_t metadata_size,
                                 const void *private_key, size_t key_size,
                                 void *signature, size_t *signature_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7
    if (dsmil_device255_init(MLOPS_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Verify PQC availability
    if (!dsmil_device255_pqc_available(&device255_ctx, TPM_ALG_ML_DSA_87)) {
        fprintf(stderr, "ERROR: ML-DSA-87 not available for provenance signing\n");
        return -1;
    }
    
    // Use TPM engine for secure signing
    if (dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_TPM) != 0) {
        fprintf(stderr, "WARNING: TPM engine unavailable, using hardware\n");
        dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_HARDWARE);
    }
    
    // Create provenance payload: model_path + metadata
    size_t model_path_len = strlen(model_path);
    size_t total_size = model_path_len + metadata_size;
    uint8_t *provenance_payload = malloc(total_size);
    if (!provenance_payload) {
        fprintf(stderr, "ERROR: Failed to allocate provenance payload\n");
        return -1;
    }
    
    memcpy(provenance_payload, model_path, model_path_len);
    memcpy(provenance_payload + model_path_len, provenance_metadata, metadata_size);
    
    // Hash provenance payload (SHA-384 per CNSA 2.0)
    uint8_t hash[48];
    size_t hash_len = sizeof(hash);
    
    if (dsmil_device255_hash(&device255_ctx, TPM_ALG_SHA384,
                             provenance_payload, total_size,
                             hash, &hash_len) != 0) {
        fprintf(stderr, "ERROR: Failed to hash provenance payload\n");
        free(provenance_payload);
        return -1;
    }
    
    free(provenance_payload);
    
    // Sign hash using ML-DSA-87 (CNSA 2.0)
    if (dsmil_device255_sign(&device255_ctx, TPM_ALG_ML_DSA_87,
                             private_key, key_size,
                             hash, hash_len,
                             signature, signature_size) != 0) {
        fprintf(stderr, "ERROR: Failed to sign provenance\n");
        return -1;
    }
    
    fprintf(stdout, "INFO: Model provenance signed using Device 255 (ML-DSA-87 CNSA 2.0)\n");
    
    return 0;
}

/**
 * @brief Verify model provenance signature using Device 255
 * 
 * @param model_path Path to model file
 * @param provenance_metadata Provenance metadata
 * @param metadata_size Metadata size
 * @param public_key Public key for verification
 * @param key_size Key size
 * @param signature Signature to verify
 * @param signature_size Signature size
 * @return 0 if valid, negative if invalid
 */
int dsmil_mlops_verify_provenance(const char *model_path,
                                  const void *provenance_metadata, size_t metadata_size,
                                  const void *public_key, size_t key_size,
                                  const void *signature, size_t signature_size) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 7
    if (dsmil_device255_init(MLOPS_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Verify PQC availability
    if (!dsmil_device255_pqc_available(&device255_ctx, TPM_ALG_ML_DSA_87)) {
        fprintf(stderr, "ERROR: ML-DSA-87 not available\n");
        return -1;
    }
    
    // Recreate provenance payload
    size_t model_path_len = strlen(model_path);
    size_t total_size = model_path_len + metadata_size;
    uint8_t *provenance_payload = malloc(total_size);
    if (!provenance_payload) {
        fprintf(stderr, "ERROR: Failed to allocate provenance payload\n");
        return -1;
    }
    
    memcpy(provenance_payload, model_path, model_path_len);
    memcpy(provenance_payload + model_path_len, provenance_metadata, metadata_size);
    
    // Hash provenance payload (SHA-384)
    uint8_t hash[48];
    size_t hash_len = sizeof(hash);
    
    if (dsmil_device255_hash(&device255_ctx, TPM_ALG_SHA384,
                             provenance_payload, total_size,
                             hash, &hash_len) != 0) {
        fprintf(stderr, "ERROR: Failed to hash provenance payload\n");
        free(provenance_payload);
        return -1;
    }
    
    free(provenance_payload);
    
    // Verify signature
    if (dsmil_device255_verify(&device255_ctx, TPM_ALG_ML_DSA_87,
                               public_key, key_size,
                               hash, hash_len,
                               signature, signature_size) != 0) {
        fprintf(stderr, "ERROR: Provenance signature verification failed\n");
        return -1;
    }
    
    fprintf(stdout, "INFO: Model provenance verified using Device 255\n");
    
    return 0;
}
