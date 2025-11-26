/**
 * @file bad_variable_time.c
 * @brief Test case for VIOLATING constant-time (variable-time instructions)
 *
 * This file demonstrates INCORRECT use of secrets with variable-time
 * instructions like division, modulo, and variable shifts.
 *
 * EXPECTED: dsmil-ct-check should report violations
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include "../../include/dsmil_attributes.h"

/**
 * VIOLATION: Division with secret operand
 * Division timing depends on operand values on many architectures
 */
DSMIL_SECRET
DSMIL_LAYER(8)
uint32_t bad_modular_reduction(uint32_t secret_value, uint32_t modulus) {
    // VIOLATION: Modulo operation on secret
    // Timing varies based on dividend/divisor
    return secret_value % modulus;  // VARIABLE_TIME violation
}

/**
 * VIOLATION: Division in RSA
 */
DSMIL_SECRET
uint64_t bad_rsa_divide(uint64_t secret_numerator, uint64_t divisor) {
    // VIOLATION: Division with secret numerator
    return secret_numerator / divisor;  // VARIABLE_TIME violation
}

/**
 * VIOLATION: Variable-shift with secret shift amount
 */
DSMIL_SECRET
uint32_t bad_variable_shift(uint32_t value, uint8_t secret_shift_count) {
    // VIOLATION: Shifting by secret amount
    // Shift timing may depend on shift count on some architectures
    return value << secret_shift_count;  // VARIABLE_TIME violation (maybe)
}

/**
 * VIOLATION: Modular exponentiation (naive, timing-vulnerable)
 */
DSMIL_SECRET
DSMIL_LAYER(8)
uint64_t bad_modular_exp(uint64_t base, uint64_t secret_exponent, uint64_t modulus) {
    uint64_t result = 1;

    // VIOLATION: Multiple issues here
    for (uint64_t i = 0; i < secret_exponent; i++) {  // SECRET_BRANCH (loop bound)
        result = (result * base) % modulus;  // VARIABLE_TIME (modulo)
    }

    return result;
}

/**
 * VIOLATION: Hash table lookup with secret-dependent modulo
 */
DSMIL_SECRET
size_t bad_hash_index(uint32_t secret_key, size_t table_size) {
    // VIOLATION: Modulo with secret operand
    return secret_key % table_size;  // VARIABLE_TIME violation
}

/**
 * VIOLATION: Bit manipulation with variable timing
 */
DSMIL_SECRET
int bad_count_set_bits(uint32_t secret_value) {
    int count = 0;

    // VIOLATION: Loop iteration depends on secret value
    while (secret_value) {  // SECRET_BRANCH violation
        count++;
        secret_value &= secret_value - 1;
    }

    return count;
}

/**
 * VIOLATION: GCD computation (multiple violations)
 */
DSMIL_SECRET
uint32_t bad_gcd(uint32_t secret_a, uint32_t secret_b) {
    // VIOLATION: Multiple issues
    while (secret_b != 0) {  // SECRET_BRANCH violation
        uint32_t temp = secret_a % secret_b;  // VARIABLE_TIME violation
        secret_a = secret_b;
        secret_b = temp;
    }
    return secret_a;
}
