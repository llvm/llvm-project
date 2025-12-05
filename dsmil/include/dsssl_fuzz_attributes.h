/**
 * @file dsssl_fuzz_attributes.h
 * @brief DSSSL Fuzzing Attribute Macros
 *
 * Provides convenient macros for annotating DSSSL code with fuzzing
 * instrumentation hints.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSSSL_FUZZ_ATTRIBUTES_H
#define DSSSL_FUZZ_ATTRIBUTES_H

/**
 * @defgroup DSSSL_FUZZ_ATTRIBUTES Fuzzing Attributes
 * @{
 */

/**
 * Mark function as part of a state machine
 *
 * @param sm_name State machine name (e.g., "tls_handshake", "ticket_lifecycle")
 *
 * Example:
 * @code
 * DSSSL_STATE_MACHINE("tls_handshake")
 * int tls_process_handshake(SSL *ssl, const uint8_t *data, size_t len) {
 *     // State machine instrumentation enabled
 * }
 * @endcode
 */
#define DSSSL_STATE_MACHINE(sm_name) \
    __attribute__((annotate("dsssl.state_machine=" #sm_name)))

/**
 * Mark function as crypto operation
 *
 * @param op_name Operation name (e.g., "ecdsa_sign", "aes_gcm_encrypt")
 *
 * Example:
 * @code
 * DSSSL_CRYPTO("ecdsa_sign")
 * int ecdsa_sign(const EC_KEY *key, uint8_t *sig, size_t *sig_len,
 *                const uint8_t *msg, size_t msg_len) {
 *     // Crypto metric instrumentation enabled
 * }
 * @endcode
 */
#define DSSSL_CRYPTO(op_name) \
    __attribute__((annotate("dsssl.crypto=" #op_name)))

/**
 * Mark loop as constant-time critical
 *
 * Example:
 * @code
 * DSSSL_CONSTANT_TIME_LOOP
 * for (size_t i = 0; i < len; i++) {
 *     // Loop iteration count tracked
 * }
 * @endcode
 */
#define DSSSL_CONSTANT_TIME_LOOP \
    __attribute__((annotate("dsssl.constant_time_loop")))

/**
 * Mark function for API misuse detection
 *
 * @param api_name API name (e.g., "AEAD_init", "cert_verify")
 *
 * Example:
 * @code
 * DSSSL_API_MISUSE_CHECK("AEAD_init")
 * int aead_init(EVP_AEAD_CTX *ctx, const EVP_AEAD *aead,
 *               const uint8_t *key, size_t key_len) {
 *     // API misuse checks enabled
 * }
 * @endcode
 */
#define DSSSL_API_MISUSE_CHECK(api_name) \
    __attribute__((annotate("dsssl.api_misuse=" #api_name)))

/**
 * Mark function for coverage instrumentation
 *
 * Example:
 * @code
 * DSSSL_COVERAGE
 * void tls_process_record(SSL *ssl, const uint8_t *record) {
 *     // Coverage instrumentation enabled
 * }
 * @endcode
 */
#define DSSSL_COVERAGE \
    __attribute__((annotate("dsssl.coverage")))

/** @} */

#endif /* DSSSL_FUZZ_ATTRIBUTES_H */
