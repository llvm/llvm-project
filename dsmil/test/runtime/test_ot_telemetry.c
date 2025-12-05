/**
 * @file test_ot_telemetry.c
 * @brief Unit tests for OT Telemetry Runtime
 *
 * Comprehensive tests for dsmil_ot_telemetry.c covering:
 * - Initialization and shutdown
 * - Event logging
 * - Safety signal updates
 * - Environment variable handling
 * - Ring buffer operations
 * - Error cases
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>

// Test utilities
static int test_passed = 0;
static int test_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (cond) { \
            test_passed++; \
            printf("PASS: %s\n", msg); \
        } else { \
            test_failed++; \
            printf("FAIL: %s\n", msg); \
        } \
    } while (0)

// Capture stderr output
static char captured_stderr[8192];
static size_t captured_len = 0;
static int stderr_fd = -1;
static int stderr_backup = -1;

static void capture_stderr_start(void) {
    fflush(stderr);
    stderr_backup = dup(STDERR_FILENO);
    stderr_fd = open("/tmp/dsmil_test_stderr", O_RDWR | O_CREAT | O_TRUNC, 0644);
    dup2(stderr_fd, STDERR_FILENO);
    captured_len = 0;
    memset(captured_stderr, 0, sizeof(captured_stderr));
}

static void capture_stderr_stop(void) {
    fflush(stderr);
    lseek(stderr_fd, 0, SEEK_SET);
    captured_len = read(stderr_fd, captured_stderr, sizeof(captured_stderr) - 1);
    close(stderr_fd);
    dup2(stderr_backup, STDERR_FILENO);
    close(stderr_backup);
    unlink("/tmp/dsmil_test_stderr");
}

// Test 1: Basic initialization
static void test_init(void) {
    printf("\n=== Test 1: Basic Initialization ===\n");
    
    int ret = dsmil_ot_telemetry_init();
    TEST_ASSERT(ret == 0, "dsmil_ot_telemetry_init() returns 0");
    
    int enabled = dsmil_ot_telemetry_is_enabled();
    TEST_ASSERT(enabled == 1, "Telemetry enabled by default");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 1 complete\n");
}

// Test 2: Environment variable disable
static void test_env_disable(void) {
    printf("\n=== Test 2: Environment Variable Disable ===\n");
    
    setenv("DSMIL_OT_TELEMETRY", "0", 1);
    
    // Reset state by calling shutdown first
    dsmil_ot_telemetry_shutdown();
    
    int ret = dsmil_ot_telemetry_init();
    TEST_ASSERT(ret == 0, "Init succeeds even when disabled");
    
    int enabled = dsmil_ot_telemetry_is_enabled();
    TEST_ASSERT(enabled == 0, "Telemetry disabled when DSMIL_OT_TELEMETRY=0");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    printf("Test 2 complete\n");
}

// Test 3: Environment variable enable
static void test_env_enable(void) {
    printf("\n=== Test 3: Environment Variable Enable ===\n");
    
    setenv("DSMIL_OT_TELEMETRY", "1", 1);
    dsmil_ot_telemetry_shutdown();
    
    int ret = dsmil_ot_telemetry_init();
    TEST_ASSERT(ret == 0, "Init succeeds");
    
    int enabled = dsmil_ot_telemetry_is_enabled();
    TEST_ASSERT(enabled == 1, "Telemetry enabled when DSMIL_OT_TELEMETRY=1");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    printf("Test 3 complete\n");
}

// Test 4: Basic event logging
static void test_event_logging(void) {
    printf("\n=== Test 4: Basic Event Logging ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "test_module",
        .func_id = "test_function",
        .file = "test.c",
        .line = 42,
        .layer = 3,
        .device = 12,
        .stage = "control",
        .mission_profile = "ics_ops",
        .authority_tier = 1,
        .build_id = 0x12345678,
        .provenance_id = 0xabcdef00,
        .signal_name = NULL,
        .signal_value = 0.0,
        .signal_min = 0.0,
        .signal_max = 0.0
    };
    
    dsmil_telemetry_event(&ev);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "Event logged to stderr");
    TEST_ASSERT(strstr(captured_stderr, "ot_path_entry") != NULL, "Event type in output");
    TEST_ASSERT(strstr(captured_stderr, "test_module") != NULL, "Module ID in output");
    TEST_ASSERT(strstr(captured_stderr, "test_function") != NULL, "Function ID in output");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 4 complete\n");
}

// Test 5: All event types
static void test_all_event_types(void) {
    printf("\n=== Test 5: All Event Types ===\n");
    
    dsmil_ot_telemetry_init();
    
    dsmil_telemetry_event_type_t types[] = {
        DSMIL_TELEMETRY_OT_PATH_ENTRY,
        DSMIL_TELEMETRY_OT_PATH_EXIT,
        DSMIL_TELEMETRY_SES_INTENT,
        DSMIL_TELEMETRY_SES_ACCEPT,
        DSMIL_TELEMETRY_SES_REJECT,
        DSMIL_TELEMETRY_INVARIANT_HIT,
        DSMIL_TELEMETRY_INVARIANT_FAIL,
        DSMIL_TELEMETRY_SS7_MSG_RX,
        DSMIL_TELEMETRY_SS7_MSG_TX,
        DSMIL_TELEMETRY_SIGTRAN_MSG_RX,
        DSMIL_TELEMETRY_SIGTRAN_MSG_TX,
        DSMIL_TELEMETRY_SIG_ANOMALY
    };
    
    const char *expected_strings[] = {
        "ot_path_entry",
        "ot_path_exit",
        "ses_intent",
        "ses_accept",
        "ses_reject",
        "invariant_hit",
        "invariant_fail",
        "ss7_msg_rx",
        "ss7_msg_tx",
        "sigtran_msg_rx",
        "sigtran_msg_tx",
        "sig_anomaly"
    };
    
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        capture_stderr_start();
        
        dsmil_telemetry_event_t ev = {
            .event_type = types[i],
            .module_id = "test",
            .func_id = "test",
            .file = "test.c",
            .line = 1,
            .layer = 0,
            .device = 0
        };
        
        dsmil_telemetry_event(&ev);
        capture_stderr_stop();
        
        TEST_ASSERT(strstr(captured_stderr, expected_strings[i]) != NULL,
                   expected_strings[i]);
    }
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 5 complete\n");
}

// Test 6: Safety signal update
static void test_safety_signal_update(void) {
    printf("\n=== Test 6: Safety Signal Update ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_INVARIANT_HIT,
        .signal_name = "line7_pressure_setpoint",
        .signal_value = 125.5,
        .signal_min = 50.0,
        .signal_max = 200.0,
        .layer = 3,
        .device = 12,
        .file = "pump.c",
        .line = 67
    };
    
    dsmil_telemetry_safety_signal_update(&ev);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "Signal update logged");
    TEST_ASSERT(strstr(captured_stderr, "line7_pressure_setpoint") != NULL, "Signal name in output");
    TEST_ASSERT(strstr(captured_stderr, "125.5") != NULL, "Signal value in output");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 6 complete\n");
}

// Test 7: Null event handling
static void test_null_event(void) {
    printf("\n=== Test 7: Null Event Handling ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event(NULL);
    dsmil_telemetry_safety_signal_update(NULL);
    
    capture_stderr_stop();
    
    // Should not crash and should not log anything
    TEST_ASSERT(captured_len == 0, "Null events don't log");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 7 complete\n");
}

// Test 8: Safety signal without name
static void test_safety_signal_no_name(void) {
    printf("\n=== Test 8: Safety Signal Without Name ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_INVARIANT_HIT,
        .signal_name = NULL,  // No signal name
        .signal_value = 125.5
    };
    
    dsmil_telemetry_safety_signal_update(&ev);
    
    capture_stderr_stop();
    
    // Should not log if signal_name is NULL
    TEST_ASSERT(captured_len == 0, "No logging when signal_name is NULL");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 8 complete\n");
}

// Test 9: Event with NULL strings
static void test_event_null_strings(void) {
    printf("\n=== Test 9: Event with NULL Strings ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = NULL,
        .func_id = NULL,
        .file = NULL,
        .line = 0,
        .layer = 0,
        .device = 0
    };
    
    dsmil_telemetry_event(&ev);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "Event logged even with NULL strings");
    TEST_ASSERT(strstr(captured_stderr, "unknown") != NULL, "NULL strings replaced with 'unknown'");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 9 complete\n");
}

// Test 10: Multiple events
static void test_multiple_events(void) {
    printf("\n=== Test 10: Multiple Events ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    for (int i = 0; i < 10; i++) {
        dsmil_telemetry_event_t ev = {
            .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
            .module_id = "test_module",
            .func_id = "test_function",
            .file = "test.c",
            .line = i,
            .layer = 3,
            .device = 12
        };
        dsmil_telemetry_event(&ev);
    }
    
    capture_stderr_stop();
    
    // Count occurrences of "ot_path_entry"
    int count = 0;
    const char *p = captured_stderr;
    while ((p = strstr(p, "ot_path_entry")) != NULL) {
        count++;
        p++;
    }
    
    TEST_ASSERT(count == 10, "All 10 events logged");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 10 complete\n");
}

// Test 11: Disabled telemetry
static void test_disabled_telemetry(void) {
    printf("\n=== Test 11: Disabled Telemetry ===\n");
    
    setenv("DSMIL_OT_TELEMETRY", "0", 1);
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "test",
        .func_id = "test",
        .file = "test.c",
        .line = 1
    };
    
    dsmil_telemetry_event(&ev);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len == 0, "No logging when telemetry disabled");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    printf("Test 11 complete\n");
}

// Test 12: Shutdown and reinit
static void test_shutdown_reinit(void) {
    printf("\n=== Test 12: Shutdown and Reinit ===\n");
    
    dsmil_ot_telemetry_init();
    int enabled1 = dsmil_ot_telemetry_is_enabled();
    dsmil_ot_telemetry_shutdown();
    
    dsmil_ot_telemetry_init();
    int enabled2 = dsmil_ot_telemetry_is_enabled();
    
    TEST_ASSERT(enabled1 == enabled2, "State consistent after shutdown/reinit");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 12 complete\n");
}

// Test 13: Event with all fields populated
static void test_complete_event(void) {
    printf("\n=== Test 13: Complete Event ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "pump_controller",
        .func_id = "pump_control_update",
        .file = "pump.c",
        .line = 42,
        .layer = 3,
        .device = 12,
        .stage = "control",
        .mission_profile = "ics_ops",
        .authority_tier = 1,
        .build_id = 0x123456789abcdef0ULL,
        .provenance_id = 0xfedcba9876543210ULL,
        .signal_name = "pressure",
        .signal_value = 125.5,
        .signal_min = 50.0,
        .signal_max = 200.0,
        .telecom_stack = "ss7",
        .ss7_role = "STP",
        .sigtran_role = "SG",
        .telecom_env = "lab",
        .telecom_if = "m3ua",
        .telecom_ep = "upstream",
        .ss7_opc = 0x1234,
        .ss7_dpc = 0x5678,
        .ss7_sio = 0x08,
        .sigtran_rctx = 0xabcd,
        .ss7_msg_class = 1,
        .ss7_msg_type = 2
    };
    
    dsmil_telemetry_event(&ev);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "Complete event logged");
    TEST_ASSERT(strstr(captured_stderr, "pump_controller") != NULL, "Module ID");
    TEST_ASSERT(strstr(captured_stderr, "ics_ops") != NULL, "Mission profile");
    TEST_ASSERT(strstr(captured_stderr, "pressure") != NULL, "Signal name");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 13 complete\n");
}

int main(void) {
    printf("========================================\n");
    printf("OT Telemetry Runtime Test Suite\n");
    printf("========================================\n");
    
    test_init();
    test_env_disable();
    test_env_enable();
    test_event_logging();
    test_all_event_types();
    test_safety_signal_update();
    test_null_event();
    test_safety_signal_no_name();
    test_event_null_strings();
    test_multiple_events();
    test_disabled_telemetry();
    test_shutdown_reinit();
    test_complete_event();
    
    printf("\n========================================\n");
    printf("Test Results: %d passed, %d failed\n", test_passed, test_failed);
    printf("========================================\n");
    
    return test_failed > 0 ? 1 : 0;
}
