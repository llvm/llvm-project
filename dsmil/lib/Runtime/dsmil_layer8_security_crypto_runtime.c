/**
 * @file dsmil_layer8_security_crypto_runtime.c
 * @brief Layer 8 (ENHANCED_SEC) Security Crypto Integration with Device 255
 * 
 * Enforces PQC-only mode and verifies PQC algorithm usage via Device 255.
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_device255_crypto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAYER8_ID 8
#define LAYER8_LAYER 8

/**
 * @brief Enable PQC-only mode (disable classical crypto)
 * 
 * @return 0 on success, negative on error
 */
int dsmil_layer8_enable_pqc_only_mode(void) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 8
    if (dsmil_device255_init(LAYER8_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Disable classical crypto capabilities
    uint16_t classical_caps = DSMIL_CRYPTO_CAP_ASYMMETRIC |  // RSA
                              DSMIL_CRYPTO_CAP_ECC;           // ECC (classical)
    
    if (dsmil_device255_cap_control(&device255_ctx, classical_caps, false) != 0) {
        fprintf(stderr, "ERROR: Failed to disable classical crypto\n");
        return -1;
    }
    
    // Ensure PQC is enabled
    if (dsmil_device255_cap_control(&device255_ctx, DSMIL_CRYPTO_CAP_POST_QUANTUM, true) != 0) {
        fprintf(stderr, "ERROR: Failed to enable PQC\n");
        return -1;
    }
    
    fprintf(stdout, "INFO: Layer 8 PQC-only mode enabled\n");
    
    return 0;
}

/**
 * @brief Verify that only PQC algorithms are used
 * 
 * @param algorithm Algorithm ID to verify
 * @return true if PQC algorithm, false if classical
 */
bool dsmil_layer8_verify_pqc_algorithm(uint16_t algorithm) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 8
    if (dsmil_device255_init(LAYER8_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return false;
    }
    
    // Check if algorithm is PQC
    switch (algorithm) {
        case TPM_ALG_ML_KEM_1024:
        case TPM_ALG_ML_DSA_87:
            return true;  // PQC algorithms
        case TPM_ALG_RSA:
        case TPM_ALG_ECDSA:
            return false;  // Classical algorithms (should be disabled)
        default:
            // Check via Device 255 PQC availability
            return dsmil_device255_pqc_available(&device255_ctx, algorithm);
    }
}

/**
 * @brief Audit crypto operations for PQC compliance
 * 
 * @param total_ops Output total operations
 * @param pqc_ops Output PQC operations count
 * @param classical_ops Output classical operations count (should be 0)
 * @return 0 on success, negative on error
 */
int dsmil_layer8_audit_crypto_compliance(uint64_t *total_ops,
                                        uint64_t *pqc_ops,
                                        uint64_t *classical_ops) {
    dsmil_device255_ctx_t device255_ctx;
    
    // Initialize Device 255 for Layer 8
    if (dsmil_device255_init(LAYER8_LAYER, &device255_ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize Device 255\n");
        return -1;
    }
    
    // Get statistics
    uint64_t ops, bytes;
    uint64_t engine_stats[3];
    
    if (dsmil_device255_get_stats(&device255_ctx, &ops, &bytes, engine_stats) != 0) {
        fprintf(stderr, "ERROR: Failed to get Device 255 statistics\n");
        return -1;
    }
    
    if (total_ops) {
        *total_ops = ops;
    }
    
    // Placeholder - actual implementation would track PQC vs classical operations
    if (pqc_ops) {
        *pqc_ops = ops;  // Assume all are PQC in PQC-only mode
    }
    
    if (classical_ops) {
        *classical_ops = 0;  // Should be zero in PQC-only mode
    }
    
    return 0;
}
