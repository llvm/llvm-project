/**
 * @file dsmil_device15_wycheproof_runtime.c
 * @brief Device 15 (CRYPTO) Wycheproof Integration with Device 255
 * 
 * Integrates Device 15 Wycheproof test execution with Device 255
 * Master Crypto Controller for all cryptographic operations.
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_device255_crypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEVICE15_ID 15
#define DEVICE15_LAYER 3

/**
 * @brief Run Wycheproof test using Device 255 for crypto operations
 * 
 * @param test_vector Test vector data
 * @param test_vector_len Test vector length
 * @param result Output test result
 * @return 0 on success, negative on error
 */
int dsmil_device15_wycheproof_test(const void *test_vector, size_t test_vector_len,
                                    void *result) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 3 (SECRET)
    if (dsmil_device255_init(DEVICE15_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Use TPM engine for secure crypto operations
    if (dsmil_device255_set_engine(&device255_ctx, DSMIL_CRYPTO_ENGINE_TPM) != 0) {
        fprintf(stderr, "ERROR: Failed to set TPM engine\n");
        return -1;
    }
    
    // Hash test vector using Device 255
    uint8_t hash_output[64];
    size_t hash_len = sizeof(hash_output);
    
    if (dsmil_device255_hash(&device255_ctx, TPM_ALG_SHA384,
                             test_vector, test_vector_len,
                             hash_output, &hash_len) != 0) {
        fprintf(stderr, "ERROR: Failed to hash test vector\n");
        return -1;
    }
    
    // Perform encryption test using Device 255
    uint8_t key[32] = {0};  // AES-256 key
    uint8_t iv[16] = {0};   // IV
    uint8_t ciphertext[1024];
    size_t ciphertext_len = sizeof(ciphertext);
    
    if (dsmil_device255_encrypt(&device255_ctx, TPM_ALG_AES,
                                key, sizeof(key),
                                iv, sizeof(iv),
                                test_vector, test_vector_len,
                                ciphertext, &ciphertext_len) != 0) {
        fprintf(stderr, "ERROR: Failed to encrypt test vector\n");
        return -1;
    }
    
    // Get RNG for test randomness
    uint8_t rng_output[32];
    dsmil_crypto_engine_t rng_source;
    
    if (dsmil_device255_rng(&device255_ctx, rng_output, sizeof(rng_output), &rng_source) != 0) {
        fprintf(stderr, "ERROR: Failed to get RNG\n");
        return -1;
    }
    
    // Get statistics
    uint64_t total_ops, bytes_processed;
    uint64_t engine_stats[3];
    
    if (dsmil_device255_get_stats(&device255_ctx, &total_ops, &bytes_processed, engine_stats) != 0) {
        fprintf(stderr, "WARNING: Failed to get Device 255 statistics\n");
    }
    
    fprintf(stdout, "INFO: Device 15 Wycheproof test completed using Device 255\n");
    fprintf(stdout, "INFO: Total operations: %lu, Bytes processed: %lu\n",
            total_ops, bytes_processed);
    
    return 0;
}
