/**
 * @file stealth_basic.c
 * @brief Basic stealth mode transformation test
 *
 * RUN: dsmil-clang -dsmil-stealth-mode=standard -S -emit-llvm %s -o - | \
 * RUN:   FileCheck %s --check-prefix=STANDARD
 *
 * RUN: dsmil-clang -dsmil-stealth-mode=aggressive -S -emit-llvm %s -o - | \
 * RUN:   FileCheck %s --check-prefix=AGGRESSIVE
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <dsmil_attributes.h>
#include <dsmil_telemetry.h>

// Test 1: Basic stealth attribute
// STANDARD: define {{.*}} @test_basic_stealth
// STANDARD: !dsmil.stealth ![[STEALTH_MD:[0-9]+]]
// STANDARD: ![[STEALTH_MD]] = !{!"dsmil.stealth.level", !"standard"}
DSMIL_STEALTH
void test_basic_stealth(void) {
    dsmil_counter_inc("test_counter");
}

// Test 2: Low signature with level
// AGGRESSIVE: define {{.*}} @test_low_signature_aggressive
// AGGRESSIVE: !dsmil.stealth ![[AGGRESSIVE_MD:[0-9]+]]
// AGGRESSIVE: ![[AGGRESSIVE_MD]] = !{!"dsmil.stealth.level", !"aggressive"}
DSMIL_LOW_SIGNATURE("aggressive")
void test_low_signature_aggressive(void) {
    dsmil_counter_inc("aggressive_counter");
}

// Test 3: Telemetry stripping
// AGGRESSIVE-NOT: call {{.*}} @dsmil_event_log
// AGGRESSIVE-NOT: call {{.*}} @dsmil_perf_latency
DSMIL_STEALTH
void test_telemetry_stripping(void) {
    dsmil_counter_inc("counter"); // May be stripped
    dsmil_event_log("event");     // Should be stripped
    dsmil_perf_latency("op", 100); // Should be stripped
}

// Test 4: Safety-critical preservation
// AGGRESSIVE: call {{.*}} @dsmil_counter_inc
// AGGRESSIVE: call {{.*}} @dsmil_forensic_security_event
DSMIL_SAFETY_CRITICAL("test")
DSMIL_STEALTH
void test_safety_critical_preservation(void) {
    dsmil_counter_inc("critical_counter"); // Preserved
    dsmil_forensic_security_event("event", DSMIL_EVENT_INFO, NULL); // Preserved
}

// Test 5: Constant-rate execution
// STANDARD: call {{.*}} @dsmil_get_timestamp_ns
// STANDARD: call {{.*}} @dsmil_nanosleep
DSMIL_CONSTANT_RATE
DSMIL_STEALTH
void test_constant_rate(void) {
    // Function body
    int x = 42;
    (void)x;
}

// Test 6: Jitter suppression attributes
// STANDARD: attributes {{.*}} "no-jump-tables"
DSMIL_JITTER_SUPPRESS
DSMIL_STEALTH
void test_jitter_suppression(void) {
    int x = 0;
    for (int i = 0; i < 100; i++) {
        x += i;
    }
}

// Test 7: Network stealth
DSMIL_NETWORK_STEALTH
DSMIL_STEALTH
void test_network_stealth(const char *msg) {
    // Network calls would be transformed here
    (void)msg;
}
