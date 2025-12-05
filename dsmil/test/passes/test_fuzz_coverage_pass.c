/**
 * @file test_fuzz_coverage_pass.c
 * @brief Integration tests for Fuzzing Coverage Pass
 *
 * Tests that the DsmilFuzzCoveragePass correctly instruments coverage sites.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: dsmil-clang++ -fsanitize=fuzzer -mllvm -dsmil-fuzz-coverage -mllvm -dsmil-fuzz-state-machine %s -c -o %t.o 2>&1 | FileCheck %s
// REQUIRES: dsmil

#include "dsmil/include/dsmil_fuzz_attributes.h"
#include "dsmil/include/dsmil_fuzz_telemetry.h"

// Test 1: Coverage instrumentation
DSMIL_FUZZ_COVERAGE
DSMIL_FUZZ_ENTRY_POINT
int test_coverage_function(const uint8_t *data, size_t size) {
    // CHECK: dsmil_fuzz_cov_hit
    if (size > 0 && data[0] == 'A') {
        return 1;
    }
    return 0;
}

// Test 2: State machine instrumentation
DSMIL_FUZZ_STATE_MACHINE("test_sm")
int test_state_machine(const uint8_t *data, size_t size) {
    // CHECK: dsmil_fuzz_state_transition
    int state = 0;
    for (size_t i = 0; i < size; i++) {
        if (data[i] == 'A') {
            state = 1;
        } else if (data[i] == 'B') {
            state = 2;
        }
    }
    return state;
}

// Test 3: Critical operation
DSMIL_FUZZ_CRITICAL_OP("test_parse")
int test_critical_op(const uint8_t *data, size_t size) {
    // Should be tracked for metrics
    int result = 0;
    for (size_t i = 0; i < size; i++) {
        result += data[i];
    }
    return result;
}

// Test 4: API misuse check
DSMIL_FUZZ_API_MISUSE_CHECK("buffer_write")
int test_api_misuse_check(void *buf, const void *data, size_t len) {
    // Should have misuse detection
    if (buf && data && len > 0) {
        // Simulated buffer write
        return 0;
    }
    return -1;
}

// Test 5: Constant-time loop
void test_constant_time_loop(const uint8_t *data, size_t len) {
    DSMIL_FUZZ_CONSTANT_TIME_LOOP
    for (size_t i = 0; i < len; i++) {
        // Constant-time operations
        volatile int dummy = data[i];
        (void)dummy;
    }
}

// Test 6: Non-instrumented function
int test_normal_function(int x) {
    // CHECK-NOT: dsmil_fuzz_cov_hit
    return x * 2;
}

int main(void) {
    uint8_t test_data[] = "test";
    test_coverage_function(test_data, sizeof(test_data));
    test_state_machine(test_data, sizeof(test_data));
    test_critical_op(test_data, sizeof(test_data));
    test_api_misuse_check(NULL, test_data, sizeof(test_data));
    test_constant_time_loop(test_data, sizeof(test_data));
    test_normal_function(42);
    return 0;
}
