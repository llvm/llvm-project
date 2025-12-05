/**
 * @file test_attributes.c
 * @brief Integration tests for all DSMIL attributes
 *
 * Tests that all attributes compile correctly and are recognized by passes.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: dsmil-clang -fdsmil-ot-telemetry -fdsmil-telecom-flags %s -c -o %t.o 2>&1 | FileCheck %s
// REQUIRES: dsmil

#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_fuzz_attributes.h"

// Test OT attributes
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_SAFETY_SIGNAL("test_signal")
static double test_signal = 100.0;

DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
void test_ot_function(void) {
    test_signal = 125.0;
}

// Test telecom attributes
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_SIGTRAN_ROLE("SG")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_TELECOM_INTERFACE("m3ua")
DSMIL_TELECOM_ENDPOINT("upstream")
void test_telecom_function(void) {
    // Telecom function
}

// Test fuzzing attributes
DSMIL_FUZZ_COVERAGE
DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_STATE_MACHINE("test_sm")
DSMIL_FUZZ_CRITICAL_OP("test_op")
DSMIL_FUZZ_API_MISUSE_CHECK("test_api")
int test_fuzz_function(const uint8_t *data, size_t len) {
    DSMIL_FUZZ_CONSTANT_TIME_LOOP
    for (size_t i = 0; i < len; i++) {
        // Constant-time loop
    }
    return 0;
}

// Test layer/device attributes
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_PLACEMENT(5, 23)
DSMIL_STAGE("serve")
void test_layer_device_function(void) {
    // Layer/device function
}

// Test security attributes
DSMIL_CLEARANCE(0x07070707)
DSMIL_ROE("ANALYSIS_ONLY")
DSMIL_GATEWAY
DSMIL_SANDBOX("test_sandbox")
DSMIL_UNTRUSTED_INPUT
DSMIL_SECRET
void test_security_function(const void *data, size_t len) {
    // Security function
}

int main(void) {
    test_ot_function();
    test_telecom_function();
    uint8_t test_data[] = "test";
    test_fuzz_function(test_data, sizeof(test_data));
    test_layer_device_function();
    test_security_function(test_data, sizeof(test_data));
    return 0;
}
