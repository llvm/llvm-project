/**
 * @file dsmil_fuzz_attributes.h
 * @brief DSLLVM General-Purpose Fuzzing Attribute Macros
 *
 * Provides convenient macros for annotating code with fuzzing
 * instrumentation hints. General-purpose for any target.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_FUZZ_ATTRIBUTES_H
#define DSMIL_FUZZ_ATTRIBUTES_H

/**
 * @defgroup DSMIL_FUZZ_ATTRIBUTES General Fuzzing Attributes
 * @{
 */

/**
 * Mark function as part of a state machine
 *
 * @param sm_name State machine name (e.g., "protocol_handshake", "parser_state")
 *
 * Example:
 * @code
 * DSMIL_FUZZ_STATE_MACHINE("http_parser")
 * int http_parse_request(const uint8_t *data, size_t len) {
 *     // State machine instrumentation enabled
 * }
 * @endcode
 */
#define DSMIL_FUZZ_STATE_MACHINE(sm_name) \
    __attribute__((annotate("dsmil.fuzz.state_machine=" #sm_name)))

/**
 * Mark function as critical operation (for metrics)
 *
 * @param op_name Operation name (e.g., "json_parse", "xml_validate")
 *
 * Example:
 * @code
 * DSMIL_FUZZ_CRITICAL_OP("json_parse")
 * int json_parse(const char *json_str) {
 *     // Operation metric instrumentation enabled
 * }
 * @endcode
 */
#define DSMIL_FUZZ_CRITICAL_OP(op_name) \
    __attribute__((annotate("dsmil.fuzz.critical_op=" #op_name)))

/**
 * Mark loop as constant-time critical
 *
 * Example:
 * @code
 * DSMIL_FUZZ_CONSTANT_TIME_LOOP
 * for (size_t i = 0; i < len; i++) {
 *     // Loop iteration count tracked
 * }
 * @endcode
 */
#define DSMIL_FUZZ_CONSTANT_TIME_LOOP \
    __attribute__((annotate("dsmil.fuzz.constant_time_loop")))

/**
 * Mark function for API misuse detection
 *
 * @param api_name API name (e.g., "buffer_write", "network_send")
 *
 * Example:
 * @code
 * DSMIL_FUZZ_API_MISUSE_CHECK("buffer_write")
 * int buffer_write(void *buf, const void *data, size_t len) {
 *     // API misuse checks enabled
 * }
 * @endcode
 */
#define DSMIL_FUZZ_API_MISUSE_CHECK(api_name) \
    __attribute__((annotate("dsmil.fuzz.api_misuse=" #api_name)))

/**
 * Mark function for coverage instrumentation
 *
 * Example:
 * @code
 * DSMIL_FUZZ_COVERAGE
 * void process_data(const uint8_t *data, size_t len) {
 *     // Coverage instrumentation enabled
 * }
 * @endcode
 */
#define DSMIL_FUZZ_COVERAGE \
    __attribute__((annotate("dsmil.fuzz.coverage")))

/**
 * Mark function as fuzzing entry point
 *
 * Example:
 * @code
 * DSMIL_FUZZ_ENTRY_POINT
 * int parse_input(const uint8_t *data, size_t len) {
 *     // Marked as primary fuzzing target
 * }
 * @endcode
 */
#define DSMIL_FUZZ_ENTRY_POINT \
    __attribute__((annotate("dsmil.fuzz.entry_point")))

/** @} */

#endif /* DSMIL_FUZZ_ATTRIBUTES_H */
