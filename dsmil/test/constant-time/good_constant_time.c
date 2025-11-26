/**
 * @file good_constant_time.c
 * @brief Test case for correct constant-time cryptographic code
 *
 * This file demonstrates proper use of DSMIL_SECRET attribute for
 * constant-time execution. All operations on secret data use only
 * constant-time primitives.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include "../../include/dsmil_attributes.h"

/**
 * Constant-time memory comparison
 * Returns 0 if equal, non-zero if different
 */
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
int crypto_memcmp(const uint8_t *a, const uint8_t *b, size_t len) {
    int result = 0;

    // Constant-time comparison: always iterate through all bytes
    // Use bitwise OR to accumulate differences without branching
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }

    return result;
}

/**
 * Constant-time conditional move (select)
 * Returns a if condition is true, b otherwise
 * This is OK because we use arithmetic, not branching
 */
DSMIL_SECRET
DSMIL_LAYER(8)
uint32_t constant_time_select(uint32_t condition, uint32_t a, uint32_t b) {
    // Convert condition to mask: 0xFFFFFFFF if true, 0x00000000 if false
    uint32_t mask = -(condition != 0);

    // Select using bitwise operations (constant-time)
    return (a & mask) | (b & ~mask);
}

/**
 * Constant-time byte lookup using masked operations
 */
DSMIL_SECRET
DSMIL_LAYER(8)
uint8_t constant_time_lookup_byte(const uint8_t *table, size_t index, size_t table_size) {
    uint8_t result = 0;

    // Access all entries, accumulating the one we want
    for (size_t i = 0; i < table_size; i++) {
        // Create mask: 0xFF if i == index, 0x00 otherwise
        uint8_t mask = -((uint8_t)(i == index));
        result |= table[i] & mask;
    }

    return result;
}

/**
 * AES-like S-box lookup (simplified, constant-time)
 */
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
void aes_sbox_substitute(const uint8_t *input, uint8_t *output, size_t len) {
    // Simplified S-box (in real AES, this is a 256-byte table)
    static const uint8_t sbox[256] = {
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        // ... (rest of S-box omitted for brevity)
    };

    for (size_t i = 0; i < len; i++) {
        // Constant-time S-box lookup
        output[i] = constant_time_lookup_byte(sbox, input[i], 256);
    }
}

/**
 * HMAC-like key mixing (simplified)
 */
DSMIL_SECRET
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
DSMIL_SAFETY_CRITICAL("crypto")
void hmac_mix_key(DSMIL_SECRET const uint8_t *key,
                   size_t key_len,
                   const uint8_t *message,
                   size_t msg_len,
                   uint8_t *output) {
    // Simple XOR mixing (constant-time)
    for (size_t i = 0; i < msg_len; i++) {
        output[i] = message[i] ^ key[i % key_len];
    }
}

/**
 * Constant-time conditional swap
 */
DSMIL_SECRET
void constant_time_swap(uint32_t condition, uint32_t *a, uint32_t *b) {
    uint32_t mask = -(condition != 0);
    uint32_t temp = mask & (*a ^ *b);
    *a ^= temp;
    *b ^= temp;
}

/**
 * Main test function
 */
DSMIL_LAYER(8)
DSMIL_DEVICE(80)
DSMIL_SANDBOX("crypto_worker")
int main(void) {
    uint8_t key1[32] = {0};
    uint8_t key2[32] = {0};
    uint8_t message[32] = {0};
    uint8_t output[32] = {0};

    // Test constant-time operations
    int cmp_result = crypto_memcmp(key1, key2, 32);

    hmac_mix_key(key1, 32, message, 32, output);

    uint32_t a = 0x12345678;
    uint32_t b = 0xABCDEF00;
    constant_time_swap(cmp_result == 0, &a, &b);

    return 0;
}
