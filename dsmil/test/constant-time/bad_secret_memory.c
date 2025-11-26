/**
 * @file bad_secret_memory.c
 * @brief Test case for VIOLATING constant-time (secret-dependent memory access)
 *
 * This file demonstrates INCORRECT use of secrets that creates timing
 * side-channels through data-dependent memory access patterns.
 *
 * EXPECTED: dsmil-ct-check should report violations
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include "../../include/dsmil_attributes.h"

/**
 * VIOLATION: Secret-dependent array indexing
 * Cache timing can reveal the index value
 */
DSMIL_SECRET
DSMIL_LAYER(8)
uint8_t bad_sbox_lookup(const uint8_t *sbox, uint8_t secret_index) {
    // VIOLATION: Indexing array with secret value
    // Cache timing leaks the index
    return sbox[secret_index];  // SECRET_MEMORY violation
}

/**
 * VIOLATION: Table lookup with secret index
 */
DSMIL_SECRET
void bad_key_schedule(const uint8_t *key, uint8_t *schedule, size_t rounds) {
    static const uint8_t round_constants[10] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
    };

    for (size_t i = 0; i < rounds; i++) {
        // VIOLATION: Using key byte as index
        uint8_t rc = round_constants[key[i] % 10];  // SECRET_MEMORY violation
        schedule[i] = rc ^ key[i];
    }
}

/**
 * VIOLATION: Secret-dependent pointer arithmetic
 */
DSMIL_SECRET
void bad_permutation_apply(const uint8_t *perm_table, const uint8_t *input,
                            uint8_t *output, size_t len) {
    for (size_t i = 0; i < len; i++) {
        // VIOLATION: Using secret input to compute address
        size_t index = input[i];  // input is secret
        output[i] = perm_table[index];  // SECRET_MEMORY violation
    }
}

/**
 * VIOLATION: Secret-dependent GEP (GetElementPtr)
 */
DSMIL_SECRET
struct KeyData {
    uint8_t bytes[32];
    uint32_t version;
};

void bad_struct_access(struct KeyData *keys, uint8_t secret_key_id, uint8_t *output) {
    // VIOLATION: Accessing array of structs with secret index
    struct KeyData *selected = &keys[secret_key_id];  // SECRET_MEMORY violation
    for (int i = 0; i < 32; i++) {
        output[i] = selected->bytes[i];
    }
}

/**
 * VIOLATION: Secret-dependent scatter/gather
 */
DSMIL_SECRET
void bad_scatter(const uint8_t *secret_indices, const uint8_t *values,
                  uint8_t *output, size_t count) {
    for (size_t i = 0; i < count; i++) {
        // VIOLATION: Storing to addresses computed from secrets
        size_t idx = secret_indices[i];
        output[idx] = values[i];  // SECRET_MEMORY violation
    }
}
