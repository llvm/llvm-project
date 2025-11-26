/**
 * @file bad_secret_branch.c
 * @brief Test case for VIOLATING constant-time requirements (secret branches)
 *
 * This file demonstrates INCORRECT use of secrets that creates timing
 * side-channels through data-dependent branching.
 *
 * EXPECTED: dsmil-ct-check should report violations
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include "../../include/dsmil_attributes.h"

/**
 * VIOLATION: Secret-dependent branch
 * Branching on secret key value leaks timing information
 */
DSMIL_SECRET
DSMIL_LAYER(8)
int bad_memcmp_early_exit(const uint8_t *a, const uint8_t *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        // VIOLATION: Early return on secret data comparison
        // Timing reveals position of first mismatch
        if (a[i] != b[i]) {
            return -1;  // SECRET_BRANCH violation
        }
    }
    return 0;
}

/**
 * VIOLATION: Secret-dependent switch
 */
DSMIL_SECRET
DSMIL_LAYER(8)
int bad_key_type_dispatch(uint8_t key_type) {
    // VIOLATION: Switching on secret key type
    // Different cases have different timing
    switch (key_type) {
        case 0: return process_rsa_key();      // SECRET_BRANCH violation
        case 1: return process_ecdsa_key();
        case 2: return process_ml_dsa_key();
        default: return -1;
    }
}

/**
 * VIOLATION: Secret-dependent if-else
 */
DSMIL_SECRET
void bad_conditional_crypto(const uint8_t *key, uint8_t *output, int use_aes) {
    // VIOLATION: Branching based on secret-derived condition
    if (use_aes) {  // SECRET_BRANCH violation if use_aes derived from key
        aes_encrypt(key, output);
    } else {
        chacha20_encrypt(key, output);
    }
}

/**
 * VIOLATION: Ternary operator on secret (becomes select instruction)
 */
DSMIL_SECRET
uint32_t bad_conditional_select(uint32_t secret_condition, uint32_t a, uint32_t b) {
    // VIOLATION: Select based on secret condition
    // While select itself is constant-time, we still flag it as potential issue
    return secret_condition ? a : b;  // SECRET_BRANCH violation (select)
}

int process_rsa_key(void) { return 1; }
int process_ecdsa_key(void) { return 2; }
int process_ml_dsa_key(void) { return 3; }
void aes_encrypt(const uint8_t *key, uint8_t *out) {}
void chacha20_encrypt(const uint8_t *key, uint8_t *out) {}
