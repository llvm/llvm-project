/**
 * @file test_telemetry_expansion_pass.c
 * @brief Integration tests for Telemetry Expansion LLVM Pass
 *
 * Tests that the DsmilTelemetryPass correctly instruments code with:
 * - New generic annotations (NET_IO, CRYPTO, PROCESS, FILE, UNTRUSTED, ERROR_HANDLER)
 * - Telemetry level-based instrumentation (normal/debug/trace)
 * - Error handler instrumentation with panic detection
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: dsmil-clang -fdsmil-ot-telemetry -fdsmil-telemetry-level=normal -mllvm -dsmil-ot-telemetry %s -c -o %t.o 2>&1 | FileCheck %s --check-prefix=NORMAL
// RUN: dsmil-clang -fdsmil-ot-telemetry -fdsmil-telemetry-level=debug -mllvm -dsmil-ot-telemetry %s -c -o %t.o 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: dsmil-clang -fdsmil-ot-telemetry -fdsmil-telemetry-level=trace -mllvm -dsmil-ot-telemetry %s -c -o %t.o 2>&1 | FileCheck %s --check-prefix=TRACE
// REQUIRES: dsmil

#include "dsmil_attributes.h"
#include "dsmil_ot_telemetry.h"

// Test 1: Network I/O annotation
DSMIL_NET_IO
DSMIL_LAYER(4)
void test_net_io_function(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: category
    // NORMAL: net
}

// Test 2: Crypto annotation
DSMIL_CRYPTO
DSMIL_LAYER(3)
void test_crypto_function(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: crypto
}

// Test 3: Process annotation
DSMIL_PROCESS
DSMIL_LAYER(5)
void test_process_function(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: process
}

// Test 4: File I/O annotation
DSMIL_FILE
DSMIL_LAYER(4)
void test_file_function(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: file
}

// Test 5: Untrusted data annotation
DSMIL_UNTRUSTED
DSMIL_LAYER(7)
void test_untrusted_function(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: untrusted
}

// Test 6: Error handler annotation
DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void test_error_handler(void) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: error
}

// Test 7: Panic function (error handler with panic name)
DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void panic_handler(const char *msg) {
    // NORMAL: dsmil_telemetry_event
    // NORMAL: panic
}

// Test 8: Exit instrumentation at debug level
DSMIL_NET_IO
void test_exit_instrumentation(void) {
    // DEBUG: dsmil_telemetry_event (entry)
    // DEBUG: dsmil_telemetry_event (exit)
    // TRACE: dsmil_telemetry_event (entry)
    // TRACE: dsmil_telemetry_event (exit)
}

// Test 9: OT-critical function (should always be instrumented)
DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
void test_ot_critical(void) {
    // NORMAL: dsmil_telemetry_event
    // DEBUG: dsmil_telemetry_event
    // TRACE: dsmil_telemetry_event
}

// Test 10: Function without annotations (should not be instrumented at normal level)
void test_unannotated_function(void) {
    // NORMAL-NOT: dsmil_telemetry_event
}

int main(void) {
    test_net_io_function();
    test_crypto_function();
    test_process_function();
    test_file_function();
    test_untrusted_function();
    test_error_handler();
    panic_handler("test");
    test_exit_instrumentation();
    test_ot_critical();
    test_unannotated_function();
    return 0;
}
