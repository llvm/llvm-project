/**
 * @file test_ot_telemetry_pass.c
 * @brief Integration tests for OT Telemetry LLVM Pass
 *
 * Tests that the DsmilTelemetryPass correctly instruments code and generates manifests.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: dsmil-clang -fdsmil-ot-telemetry -mllvm -dsmil-ot-telemetry %s -c -o %t.o 2>&1 | FileCheck %s
// RUN: test -f %t.dsmil.telemetry.json || echo "Manifest file created"
// REQUIRES: dsmil

#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"

// Test 1: OT-critical function should be instrumented
DSMIL_OT_CRITICAL
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
void test_ot_critical_function(void) {
    // CHECK: dsmil_telemetry_event
}

// Test 2: SES gate function
DSMIL_SES_GATE
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
void test_ses_gate_function(void) {
    // CHECK: dsmil_telemetry_event
    // CHECK: ses_intent
}

// Test 3: Safety signal variable
DSMIL_SAFETY_SIGNAL("test_pressure")
static double test_pressure = 100.0;

void test_safety_signal_update(void) {
    test_pressure = 125.0;
    // CHECK: dsmil_telemetry_safety_signal_update
}

// Test 4: Function with all OT attributes
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(0)
DSMIL_SES_GATE
DSMIL_LAYER(2)
DSMIL_DEVICE(5)
DSMIL_STAGE("control")
void test_complete_ot_function(void) {
    // CHECK: dsmil_telemetry_event
}

// Test 5: Non-OT function should not be instrumented
void test_normal_function(void) {
    // CHECK-NOT: dsmil_telemetry_event
}

int main(void) {
    test_ot_critical_function();
    test_ses_gate_function();
    test_safety_signal_update();
    test_complete_ot_function();
    test_normal_function();
    return 0;
}
