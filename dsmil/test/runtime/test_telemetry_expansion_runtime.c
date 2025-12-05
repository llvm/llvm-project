/**
 * @file test_telemetry_expansion_runtime.c
 * @brief Unit tests for Telemetry Expansion Runtime Features
 *
 * Tests for new telemetry expansion runtime features:
 * - Telemetry level API
 * - Level gating logic
 * - New event types (30-36)
 * - New event fields (category, op, status_code, resource, error_msg, elapsed_ns)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

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
static char stderr_file_path[256];
static int stderr_backup_fd = -1;
static int stderr_file_fd = -1;

static void capture_stderr_start(void) {
    fflush(stderr);
    stderr_backup_fd = dup(STDERR_FILENO);
    snprintf(stderr_file_path, sizeof(stderr_file_path), "/tmp/dsmil_stderr_%d_%lu", getpid(), (unsigned long)time(NULL));
    stderr_file_fd = open(stderr_file_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (stderr_file_fd < 0) {
        dup2(stderr_backup_fd, STDERR_FILENO);
        return;
    }
    dup2(stderr_file_fd, STDERR_FILENO);
    setvbuf(stderr, NULL, _IONBF, 0);
    captured_len = 0;
    memset(captured_stderr, 0, sizeof(captured_stderr));
}

static void capture_stderr_stop(void) {
    fflush(stderr);
    fsync(STDERR_FILENO);
    if (stderr_backup_fd >= 0) {
        dup2(stderr_backup_fd, STDERR_FILENO);
        close(stderr_backup_fd);
        stderr_backup_fd = -1;
    }
    if (stderr_file_fd >= 0) {
        close(stderr_file_fd);
        stderr_file_fd = -1;
    }
    setvbuf(stderr, NULL, _IOLBF, 0);
    FILE *f = fopen(stderr_file_path, "r");
    if (f) {
        captured_len = fread(captured_stderr, 1, sizeof(captured_stderr) - 1, f);
        if (captured_len > 0) {
            captured_stderr[captured_len] = '\0';
        }
        fclose(f);
    }
    unlink(stderr_file_path);
}

// Test 1: Telemetry level API
static void test_telemetry_level_api(void) {
    printf("\n=== Test 1: Telemetry Level API ===\n");
    
    unsetenv("DSMIL_TELEMETRY_LEVEL");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    dsmil_telemetry_level_t level = dsmil_telemetry_get_level();
    TEST_ASSERT(level >= DSMIL_TELEMETRY_LEVEL_OFF && level <= DSMIL_TELEMETRY_LEVEL_TRACE,
                "Level is valid");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 1 complete\n");
}

// Test 2: Runtime level override
static void test_runtime_level_override(void) {
    printf("\n=== Test 2: Runtime Level Override ===\n");
    
    setenv("DSMIL_TELEMETRY_LEVEL", "debug", 1);
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    dsmil_telemetry_level_t level = dsmil_telemetry_get_level();
    TEST_ASSERT(level == DSMIL_TELEMETRY_LEVEL_DEBUG, "Runtime level override works");
    
    unsetenv("DSMIL_TELEMETRY_LEVEL");
    dsmil_ot_telemetry_shutdown();
    printf("Test 2 complete\n");
}

// Test 3: Level gating - min level
static void test_level_gating_min(void) {
    printf("\n=== Test 3: Level Gating (Min) ===\n");
    
    setenv("DSMIL_TELEMETRY_LEVEL", "min", 1);
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    // OT event should pass
    dsmil_telemetry_event_t ev_ot = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "test",
        .func_id = "test",
        .file = "test.c",
        .line = 1
    };
    dsmil_telemetry_event(&ev_ot);
    
    // Generic event should be filtered
    dsmil_telemetry_event_t ev_net = {
        .event_type = DSMIL_TELEMETRY_NET_IO,
        .category = "net",
        .module_id = "test",
        .func_id = "test",
        .file = "test.c",
        .line = 1
    };
    dsmil_telemetry_event(&ev_net);
    
    capture_stderr_stop();
    
    TEST_ASSERT(strstr(captured_stderr, "ot_path_entry") != NULL, "OT event logged");
    TEST_ASSERT(strstr(captured_stderr, "net_io") == NULL, "Generic event filtered");
    
    unsetenv("DSMIL_TELEMETRY_LEVEL");
    dsmil_ot_telemetry_shutdown();
    printf("Test 3 complete\n");
}

// Test 4: New event types
static void test_new_event_types(void) {
    printf("\n=== Test 4: New Event Types ===\n");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    dsmil_telemetry_event_type_t new_types[] = {
        DSMIL_TELEMETRY_NET_IO,
        DSMIL_TELEMETRY_CRYPTO,
        DSMIL_TELEMETRY_PROCESS,
        DSMIL_TELEMETRY_FILE,
        DSMIL_TELEMETRY_UNTRUSTED,
        DSMIL_TELEMETRY_ERROR,
        DSMIL_TELEMETRY_PANIC
    };
    
    const char *expected_strings[] = {
        "net_io",
        "crypto",
        "process",
        "file",
        "untrusted",
        "error",
        "panic"
    };
    
    for (size_t i = 0; i < sizeof(new_types) / sizeof(new_types[0]); i++) {
        capture_stderr_start();
        
        dsmil_telemetry_event_t ev = {
            .event_type = new_types[i],
            .module_id = "test",
            .func_id = "test",
            .file = "test.c",
            .line = 1,
            .category = expected_strings[i]
        };
        
        dsmil_telemetry_event(&ev);
        fflush(stderr);
        capture_stderr_stop();
        
        TEST_ASSERT(strstr(captured_stderr, expected_strings[i]) != NULL,
                   expected_strings[i]);
    }
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 4 complete\n");
}

// Test 5: New event fields
static void test_new_event_fields(void) {
    printf("\n=== Test 5: New Event Fields ===\n");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_NET_IO,
        .module_id = "test",
        .func_id = "connect",
        .file = "network.c",
        .line = 42,
        .category = "net",
        .op = "connect",
        .status_code = 0,
        .resource = "tcp://example.com:80",
        .error_msg = NULL,
        .elapsed_ns = 1234567ULL
    };
    
    dsmil_telemetry_event(&ev);
    fflush(stderr);
    capture_stderr_stop();
    
    TEST_ASSERT(strstr(captured_stderr, "net") != NULL, "Category field present");
    TEST_ASSERT(strstr(captured_stderr, "connect") != NULL, "Op field present");
    TEST_ASSERT(strstr(captured_stderr, "tcp://example.com:80") != NULL, "Resource field present");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 5 complete\n");
}

// Test 6: Error event with error message
static void test_error_event_with_message(void) {
    printf("\n=== Test 6: Error Event with Message ===\n");
    
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_ERROR,
        .module_id = "test",
        .func_id = "handle_error",
        .file = "error.c",
        .line = 100,
        .category = "error",
        .op = "error",
        .status_code = -1,
        .resource = NULL,
        .error_msg = "Connection failed",
        .elapsed_ns = 0
    };
    
    dsmil_telemetry_event(&ev);
    fflush(stderr);
    capture_stderr_stop();
    
    TEST_ASSERT(strstr(captured_stderr, "error") != NULL, "Error event logged");
    TEST_ASSERT(strstr(captured_stderr, "Connection failed") != NULL, "Error message present");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 6 complete\n");
}

// Test 7: Level allows function
static void test_level_allows(void) {
    printf("\n=== Test 7: Level Allows Function ===\n");
    
    setenv("DSMIL_TELEMETRY_LEVEL", "normal", 1);
    unsetenv("DSMIL_OT_TELEMETRY");
    dsmil_ot_telemetry_shutdown();
    dsmil_ot_telemetry_init();
    
    // Normal level should allow generic events
    int allows = dsmil_telemetry_level_allows(DSMIL_TELEMETRY_NET_IO, "net");
    TEST_ASSERT(allows == 1, "Normal level allows net events");
    
    allows = dsmil_telemetry_level_allows(DSMIL_TELEMETRY_CRYPTO, "crypto");
    TEST_ASSERT(allows == 1, "Normal level allows crypto events");
    
    unsetenv("DSMIL_TELEMETRY_LEVEL");
    dsmil_ot_telemetry_shutdown();
    printf("Test 7 complete\n");
}

int main(void) {
    printf("========================================\n");
    printf("Telemetry Expansion Runtime Test Suite\n");
    printf("========================================\n");
    
    test_telemetry_level_api();
    test_runtime_level_override();
    test_level_gating_min();
    test_new_event_types();
    test_new_event_fields();
    test_error_event_with_message();
    test_level_allows();
    
    printf("\n========================================\n");
    printf("Test Results: %d passed, %d failed\n", test_passed, test_failed);
    printf("========================================\n");
    
    return test_failed > 0 ? 1 : 0;
}
