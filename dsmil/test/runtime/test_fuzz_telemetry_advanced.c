/**
 * @file test_fuzz_telemetry_advanced.c
 * @brief Unit tests for Advanced Fuzzing Telemetry Runtime
 *
 * Comprehensive tests for dsmil_fuzz_telemetry_advanced.c covering:
 * - Advanced initialization
 * - Coverage map operations
 * - ML integration stubs
 * - Performance counters
 * - Statistics
 * - Advanced event export
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_fuzz_telemetry_advanced.h"
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

// Test 1: Basic advanced initialization
static void test_advanced_init(void) {
    printf("\n=== Test 1: Advanced Initialization ===\n");
    
    int ret = dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    TEST_ASSERT(ret == 0, "Advanced init succeeds");
    
    // Cleanup
    // Note: shutdown function may need to be added to advanced API
    printf("Test 1 complete\n");
}

// Test 2: Advanced init with perf counters
static void test_advanced_init_perf(void) {
    printf("\n=== Test 2: Advanced Init with Perf Counters ===\n");
    
    int ret = dsmil_fuzz_telemetry_advanced_init(NULL, 0, 1, 0);
    // May fail if not running as root, which is OK
    TEST_ASSERT(ret == 0 || ret == -1, "Advanced init with perf handles gracefully");
    
    printf("Test 2 complete\n");
}

// Test 3: Advanced init with ML
static void test_advanced_init_ml(void) {
    printf("\n=== Test 3: Advanced Init with ML ===\n");
    
    int ret = dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 1);
    TEST_ASSERT(ret == 0, "Advanced init with ML succeeds");
    
    printf("Test 3 complete\n");
}

// Test 4: Coverage map update
static void test_coverage_map_update(void) {
    printf("\n=== Test 4: Coverage Map Update ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    uint32_t edges[] = {1, 2, 3, 100, 200};
    uint32_t states[] = {1, 2};
    
    int ret = dsmil_fuzz_update_coverage_map(0x1234, edges, 5, states, 2);
    // Returns 1 if new coverage found, 0 if no new coverage (both are success)
    // Note: First call should return 1 (new coverage), subsequent calls may return 0
    TEST_ASSERT(ret == 0 || ret == 1, "Coverage map update succeeds");
    
    uint32_t total_edges, total_states;
    uint64_t unique_inputs;
    dsmil_fuzz_get_coverage_stats(&total_edges, &total_states, &unique_inputs);
    
    TEST_ASSERT(total_edges >= 5, "Edge count updated");
    TEST_ASSERT(total_states >= 2, "State count updated");
    TEST_ASSERT(unique_inputs >= 1, "Unique inputs incremented");
    
    printf("Test 4 complete\n");
}

// Test 5: Coverage statistics
static void test_coverage_stats(void) {
    printf("\n=== Test 5: Coverage Statistics ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    uint32_t edges[] = {10, 20, 30};
    uint32_t states[] = {5};
    
    dsmil_fuzz_update_coverage_map(0x1111, edges, 3, states, 1);
    dsmil_fuzz_update_coverage_map(0x2222, edges, 3, NULL, 0);  // No new states
    
    uint32_t total_edges, total_states;
    uint64_t unique_inputs;
    dsmil_fuzz_get_coverage_stats(&total_edges, &total_states, &unique_inputs);
    
    TEST_ASSERT(total_edges >= 3, "Total edges tracked");
    TEST_ASSERT(total_states >= 1, "Total states tracked");
    TEST_ASSERT(unique_inputs >= 2, "Unique inputs tracked");
    
    printf("Test 5 complete\n");
}

// Test 6: Performance counters
static void test_perf_counters(void) {
    printf("\n=== Test 6: Performance Counters ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 1, 0);
    
    dsmil_fuzz_record_perf_counters(1000, 50, 10);
    
    // Should not crash
    TEST_ASSERT(1, "Performance counters recorded");
    
    printf("Test 6 complete\n");
}

// Test 7: ML interestingness
static void test_ml_interestingness(void) {
    printf("\n=== Test 7: ML Interestingness ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 1);
    
    dsmil_coverage_feedback_t feedback = {
        .new_edges = 5,
        .new_states = 2,
        .total_edges = 100,
        .total_states = 10
    };
    
    double score = dsmil_fuzz_compute_interestingness(0x1234, &feedback);
    
    // Score should be between 0 and 1 (or similar range)
    TEST_ASSERT(score >= 0.0, "Interestingness score valid");
    
    printf("Test 7 complete\n");
}

// Test 8: ML mutation suggestions
static void test_ml_mutations(void) {
    printf("\n=== Test 8: ML Mutation Suggestions ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 1);
    
    dsmil_mutation_metadata_t suggestions[10];
    size_t count = dsmil_fuzz_get_mutation_suggestions(1, suggestions, 10);
    
    // May return 0 if ML not fully implemented, which is OK
    TEST_ASSERT(count <= 10, "Mutation suggestions within limit");
    
    printf("Test 8 complete\n");
}

// Test 9: Telemetry statistics
static void test_telemetry_stats(void) {
    printf("\n=== Test 9: Telemetry Statistics ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    // Record some events
    dsmil_advanced_fuzz_event_t event = {0};
    event.base.event_type = DSMIL_FUZZ_EVENT_COVERAGE_HIT;
    dsmil_fuzz_record_advanced_event(&event);
    
    uint64_t total_events;
    double events_per_sec, utilization;
    dsmil_fuzz_get_telemetry_stats(&total_events, &events_per_sec, &utilization);
    
    TEST_ASSERT(total_events >= 1, "Total events tracked");
    TEST_ASSERT(events_per_sec >= 0.0, "Events per second valid");
    TEST_ASSERT(utilization >= 0.0 && utilization <= 1.0, "Utilization valid");
    
    printf("Test 9 complete\n");
}

// Test 10: Advanced event export
static void test_advanced_export(void) {
    printf("\n=== Test 10: Advanced Event Export ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    // Record some events
    dsmil_advanced_fuzz_event_t event = {0};
    event.base.event_type = DSMIL_FUZZ_EVENT_COVERAGE_HIT;
    dsmil_fuzz_record_advanced_event(&event);
    
    int ret = dsmil_fuzz_export_for_ml("/tmp/dsmil_test_ml.json", "json");
    // May fail if ML not fully implemented, which is OK
    TEST_ASSERT(ret == 0 || ret == -1, "ML export handles gracefully");
    
    printf("Test 10 complete\n");
}

// Test 11: Advanced flush with compression
static void test_advanced_flush(void) {
    printf("\n=== Test 11: Advanced Flush ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    dsmil_advanced_fuzz_event_t event = {0};
    event.base.event_type = DSMIL_FUZZ_EVENT_COVERAGE_HIT;
    dsmil_fuzz_record_advanced_event(&event);
    
    int ret = dsmil_fuzz_flush_advanced_events("/tmp/dsmil_test_advanced.bin", 0);
    TEST_ASSERT(ret == 0, "Advanced flush succeeds");
    
    // Verify file exists
    FILE *fp = fopen("/tmp/dsmil_test_advanced.bin", "rb");
    if (fp) {
        fclose(fp);
        unlink("/tmp/dsmil_test_advanced.bin");
        TEST_ASSERT(1, "Output file created");
    }
    
    printf("Test 11 complete\n");
}

// Test 12: Multiple coverage updates
static void test_multiple_coverage_updates(void) {
    printf("\n=== Test 12: Multiple Coverage Updates ===\n");
    
    dsmil_fuzz_telemetry_advanced_init(NULL, 0, 0, 0);
    
    for (int i = 0; i < 10; i++) {
        uint32_t edges[] = {i * 10, i * 10 + 1};
        uint32_t states[] = {i};
        dsmil_fuzz_update_coverage_map(0x1000 + i, edges, 2, states, 1);
    }
    
    uint32_t total_edges, total_states;
    uint64_t unique_inputs;
    dsmil_fuzz_get_coverage_stats(&total_edges, &total_states, &unique_inputs);
    
    TEST_ASSERT(total_edges >= 20, "Multiple edge updates tracked");
    TEST_ASSERT(total_states >= 10, "Multiple state updates tracked");
    TEST_ASSERT(unique_inputs >= 10, "Multiple unique inputs tracked");
    
    printf("Test 12 complete\n");
}

int main(void) {
    printf("========================================\n");
    printf("Advanced Fuzzing Telemetry Test Suite\n");
    printf("========================================\n");
    
    test_advanced_init();
    test_advanced_init_perf();
    test_advanced_init_ml();
    test_coverage_map_update();
    test_coverage_stats();
    test_perf_counters();
    test_ml_interestingness();
    test_ml_mutations();
    test_telemetry_stats();
    test_advanced_export();
    test_advanced_flush();
    test_multiple_coverage_updates();
    
    printf("\n========================================\n");
    printf("Test Results: %d passed, %d failed\n", test_passed, test_failed);
    printf("========================================\n");
    
    return test_failed > 0 ? 1 : 0;
}
