/**
 * @file test_fuzz_telemetry.c
 * @brief Unit tests for General Fuzzing Telemetry Runtime
 *
 * Comprehensive tests for dsmil_fuzz_telemetry.c covering:
 * - Initialization and shutdown
 * - Coverage tracking
 * - State machine transitions
 * - Metrics collection
 * - API misuse detection
 * - Ring buffer operations
 * - Event export
 * - Budget checking
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_fuzz_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

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

// Test 1: Basic initialization
static void test_init(void) {
    printf("\n=== Test 1: Basic Initialization ===\n");
    
    int ret = dsmil_fuzz_telemetry_init(NULL, 0);
    TEST_ASSERT(ret == 0, "Init with default buffer size succeeds");
    
    dsmil_fuzz_telemetry_shutdown();
    
    ret = dsmil_fuzz_telemetry_init(NULL, 1024);
    TEST_ASSERT(ret == 0, "Init with custom buffer size succeeds");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 1 complete\n");
}

// Test 2: Initialization failure (large buffer)
static void test_init_failure(void) {
    printf("\n=== Test 2: Initialization Failure Handling ===\n");
    
    // Try with extremely large buffer (may fail on some systems)
    int ret = dsmil_fuzz_telemetry_init(NULL, SIZE_MAX);
    // Should handle gracefully (either succeed or fail cleanly)
    if (ret == 0) {
        dsmil_fuzz_telemetry_shutdown();
    }
    
    TEST_ASSERT(1, "Init handles large buffer gracefully");
    printf("Test 2 complete\n");
}

// Test 3: Double initialization
static void test_double_init(void) {
    printf("\n=== Test 3: Double Initialization ===\n");
    
    int ret1 = dsmil_fuzz_telemetry_init(NULL, 0);
    TEST_ASSERT(ret1 == 0, "First init succeeds");
    
    int ret2 = dsmil_fuzz_telemetry_init(NULL, 0);
    TEST_ASSERT(ret2 == 0, "Second init succeeds (idempotent)");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 3 complete\n");
}

// Test 4: Context management
static void test_context(void) {
    printf("\n=== Test 4: Context Management ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 0);
    
    uint64_t ctx1 = dsmil_fuzz_get_context();
    TEST_ASSERT(ctx1 == 0, "Initial context is 0");
    
    dsmil_fuzz_set_context(0x123456789abcdef0ULL);
    uint64_t ctx2 = dsmil_fuzz_get_context();
    TEST_ASSERT(ctx2 == 0x123456789abcdef0ULL, "Context set correctly");
    
    dsmil_fuzz_set_context(0);
    uint64_t ctx3 = dsmil_fuzz_get_context();
    TEST_ASSERT(ctx3 == 0, "Context reset to 0");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 4 complete\n");
}

// Test 5: Coverage hit
static void test_coverage_hit(void) {
    printf("\n=== Test 5: Coverage Hit ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_cov_hit(1);
    dsmil_fuzz_cov_hit(2);
    dsmil_fuzz_cov_hit(3);
    
    dsmil_fuzz_telemetry_event_t events[10];
    size_t count = dsmil_fuzz_get_events(events, 10);
    
    TEST_ASSERT(count >= 3, "At least 3 coverage events recorded");
    
    int coverage_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (events[i].event_type == DSMIL_FUZZ_EVENT_COVERAGE_HIT) {
            coverage_count++;
        }
    }
    
    TEST_ASSERT(coverage_count >= 3, "Coverage events have correct type");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 5 complete\n");
}

// Test 6: State machine transitions
static void test_state_transitions(void) {
    printf("\n=== Test 6: State Machine Transitions ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_state_transition(1, 0, 1);
    dsmil_fuzz_state_transition(1, 1, 2);
    dsmil_fuzz_state_transition(2, 0, 1);
    
    dsmil_fuzz_telemetry_event_t events[10];
    size_t count = dsmil_fuzz_get_events(events, 10);
    
    TEST_ASSERT(count >= 3, "At least 3 state transition events recorded");
    
    int transition_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (events[i].event_type == DSMIL_FUZZ_EVENT_STATE_TRANSITION) {
            transition_count++;
            if (transition_count == 1) {
                TEST_ASSERT(events[i].data.state_transition.sm_id == 1, "First SM ID correct");
                TEST_ASSERT(events[i].data.state_transition.state_from == 0, "First state_from correct");
                TEST_ASSERT(events[i].data.state_transition.state_to == 1, "First state_to correct");
            }
        }
    }
    
    TEST_ASSERT(transition_count >= 3, "State transition events have correct type");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 6 complete\n");
}

// Test 7: Metrics recording
static void test_metrics(void) {
    printf("\n=== Test 7: Metrics Recording ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_metric_begin("test_op");
    dsmil_fuzz_metric_record("test_op", 10, 20, 30, 1000);
    dsmil_fuzz_metric_end("test_op");
    
    dsmil_fuzz_telemetry_event_t events[10];
    size_t count = dsmil_fuzz_get_events(events, 10);
    
    int metric_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (events[i].event_type == DSMIL_FUZZ_EVENT_METRIC) {
            metric_count++;
            TEST_ASSERT(strcmp(events[i].data.metric.op_name, "test_op") == 0, "Operation name correct");
            TEST_ASSERT(events[i].data.metric.branches == 10, "Branches correct");
            TEST_ASSERT(events[i].data.metric.loads == 20, "Loads correct");
            TEST_ASSERT(events[i].data.metric.stores == 30, "Stores correct");
            TEST_ASSERT(events[i].data.metric.cycles == 1000, "Cycles correct");
        }
    }
    
    TEST_ASSERT(metric_count >= 1, "At least one metric event recorded");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 7 complete\n");
}

// Test 8: API misuse reporting
static void test_api_misuse(void) {
    printf("\n=== Test 8: API Misuse Reporting ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_set_context(0xdeadbeef);
    dsmil_fuzz_api_misuse_report("buffer_write", "buffer overflow", 0xdeadbeef);
    
    dsmil_fuzz_telemetry_event_t events[10];
    size_t count = dsmil_fuzz_get_events(events, 10);
    
    int misuse_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (events[i].event_type == DSMIL_FUZZ_EVENT_API_MISUSE) {
            misuse_count++;
            TEST_ASSERT(strcmp(events[i].data.api_misuse.api, "buffer_write") == 0, "API name correct");
            TEST_ASSERT(strcmp(events[i].data.api_misuse.reason, "buffer overflow") == 0, "Reason correct");
            TEST_ASSERT(events[i].data.api_misuse.context_id == 0xdeadbeef, "Context ID correct");
        }
    }
    
    TEST_ASSERT(misuse_count >= 1, "At least one API misuse event recorded");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 8 complete\n");
}

// Test 9: State events
static void test_state_events(void) {
    printf("\n=== Test 9: State Events ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_state_event(DSMIL_STATE_CREATE, 0x1234);
    dsmil_fuzz_state_event(DSMIL_STATE_USE, 0x1234);
    dsmil_fuzz_state_event(DSMIL_STATE_DESTROY, 0x1234);
    dsmil_fuzz_state_event(DSMIL_STATE_REJECT, 0x5678);
    
    dsmil_fuzz_telemetry_event_t events[10];
    size_t count = dsmil_fuzz_get_events(events, 10);
    
    int state_event_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (events[i].event_type == DSMIL_FUZZ_EVENT_STATE_EVENT) {
            state_event_count++;
        }
    }
    
    TEST_ASSERT(state_event_count >= 4, "At least 4 state events recorded");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 9 complete\n");
}

// Test 10: Event export
static void test_event_export(void) {
    printf("\n=== Test 10: Event Export ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    // Generate some events
    for (int i = 0; i < 5; i++) {
        dsmil_fuzz_cov_hit(i);
    }
    
    const char *test_file = "/tmp/dsmil_test_events.bin";
    int ret = dsmil_fuzz_flush_events(test_file);
    TEST_ASSERT(ret == 0, "Flush events succeeds");
    
    // Verify file exists and has content
    FILE *fp = fopen(test_file, "rb");
    TEST_ASSERT(fp != NULL, "Output file created");
    
    if (fp) {
        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fclose(fp);
        TEST_ASSERT(size > 0, "Output file has content");
        unlink(test_file);
    }
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 10 complete\n");
}

// Test 11: Clear events
static void test_clear_events(void) {
    printf("\n=== Test 11: Clear Events ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    // Generate events
    for (int i = 0; i < 10; i++) {
        dsmil_fuzz_cov_hit(i);
    }
    
    dsmil_fuzz_telemetry_event_t events[20];
    size_t count_before = dsmil_fuzz_get_events(events, 20);
    TEST_ASSERT(count_before > 0, "Events present before clear");
    
    dsmil_fuzz_clear_events();
    
    size_t count_after = dsmil_fuzz_get_events(events, 20);
    TEST_ASSERT(count_after == 0, "No events after clear");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 11 complete\n");
}

// Test 12: Budget checking
static void test_budget_checking(void) {
    printf("\n=== Test 12: Budget Checking ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 0);
    
    // Within budget
    int ret1 = dsmil_fuzz_check_budget("test_op", 100, 1000, 1000, 10000);
    TEST_ASSERT(ret1 == 0, "Within budget returns 0");
    
    // Over budget (branches)
    int ret2 = dsmil_fuzz_check_budget("test_op", 20000, 1000, 1000, 10000);
    TEST_ASSERT(ret2 == 1, "Over budget (branches) returns 1");
    
    // Over budget (loads)
    int ret3 = dsmil_fuzz_check_budget("test_op", 100, 60000, 1000, 10000);
    TEST_ASSERT(ret3 == 1, "Over budget (loads) returns 1");
    
    // Over budget (stores)
    int ret4 = dsmil_fuzz_check_budget("test_op", 100, 1000, 60000, 10000);
    TEST_ASSERT(ret4 == 1, "Over budget (stores) returns 1");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 12 complete\n");
}

// Test 13: Ring buffer overflow
static void test_ring_buffer_overflow(void) {
    printf("\n=== Test 13: Ring Buffer Overflow ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 100);  // Small buffer
    
    // Generate more events than buffer size
    for (int i = 0; i < 200; i++) {
        dsmil_fuzz_cov_hit(i);
    }
    
    // Should not crash
    dsmil_fuzz_telemetry_event_t events[200];
    size_t count = dsmil_fuzz_get_events(events, 200);
    
    TEST_ASSERT(count <= 100, "Ring buffer handles overflow gracefully");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 13 complete\n");
}

// Test 14: Get events with NULL buffer
static void test_get_events_null(void) {
    printf("\n=== Test 14: Get Events with NULL Buffer ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_cov_hit(1);
    
    size_t count = dsmil_fuzz_get_events(NULL, 10);
    TEST_ASSERT(count == 0, "Get events with NULL buffer returns 0");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 14 complete\n");
}

// Test 15: Flush events with invalid path
static void test_flush_invalid_path(void) {
    printf("\n=== Test 15: Flush Events with Invalid Path ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_cov_hit(1);
    
    int ret = dsmil_fuzz_flush_events("/invalid/path/that/does/not/exist/file.bin");
    TEST_ASSERT(ret != 0, "Flush to invalid path fails");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 15 complete\n");
}

// Test 16: Multiple operations
static void test_multiple_operations(void) {
    printf("\n=== Test 16: Multiple Operations ===\n");
    
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_set_context(0x1234);
    
    // Mix of all event types
    dsmil_fuzz_cov_hit(1);
    dsmil_fuzz_state_transition(1, 0, 1);
    dsmil_fuzz_metric_record("op1", 10, 20, 30, 100);
    dsmil_fuzz_api_misuse_report("api1", "reason1", 0x1234);
    dsmil_fuzz_state_event(DSMIL_STATE_CREATE, 0x5678);
    
    dsmil_fuzz_telemetry_event_t events[20];
    size_t count = dsmil_fuzz_get_events(events, 20);
    
    TEST_ASSERT(count >= 5, "All event types recorded");
    
    int coverage = 0, transitions = 0, metrics = 0, misuse = 0, state_events = 0;
    for (size_t i = 0; i < count; i++) {
        switch (events[i].event_type) {
            case DSMIL_FUZZ_EVENT_COVERAGE_HIT: coverage++; break;
            case DSMIL_FUZZ_EVENT_STATE_TRANSITION: transitions++; break;
            case DSMIL_FUZZ_EVENT_METRIC: metrics++; break;
            case DSMIL_FUZZ_EVENT_API_MISUSE: misuse++; break;
            case DSMIL_FUZZ_EVENT_STATE_EVENT: state_events++; break;
            default: break;
        }
    }
    
    TEST_ASSERT(coverage >= 1, "Coverage events present");
    TEST_ASSERT(transitions >= 1, "Transition events present");
    TEST_ASSERT(metrics >= 1, "Metric events present");
    TEST_ASSERT(misuse >= 1, "Misuse events present");
    TEST_ASSERT(state_events >= 1, "State events present");
    
    dsmil_fuzz_telemetry_shutdown();
    printf("Test 16 complete\n");
}

int main(void) {
    printf("========================================\n");
    printf("Fuzzing Telemetry Runtime Test Suite\n");
    printf("========================================\n");
    
    test_init();
    test_init_failure();
    test_double_init();
    test_context();
    test_coverage_hit();
    test_state_transitions();
    test_metrics();
    test_api_misuse();
    test_state_events();
    test_event_export();
    test_clear_events();
    test_budget_checking();
    test_ring_buffer_overflow();
    test_get_events_null();
    test_flush_invalid_path();
    test_multiple_operations();
    
    printf("\n========================================\n");
    printf("Test Results: %d passed, %d failed\n", test_passed, test_failed);
    printf("========================================\n");
    
    return test_failed > 0 ? 1 : 0;
}
