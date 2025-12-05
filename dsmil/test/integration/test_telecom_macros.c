/**
 * @file test_telecom_macros.c
 * @brief Integration tests for Telecom Helper Macros
 *
 * Tests that telecom logging macros work correctly.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_telecom_log.h"
#include "dsmil/include/dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

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

// Test 1: SS7 RX macro
static void test_ss7_rx_macro(void) {
    printf("\n=== Test 1: SS7 RX Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SS7_RX(0x1234, 0x5678, 0x08, 1, 2);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "SS7 RX macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "ss7_msg_rx") != NULL, "Event type correct");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 1 complete\n");
}

// Test 2: SS7 TX macro
static void test_ss7_tx_macro(void) {
    printf("\n=== Test 2: SS7 TX Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SS7_TX(0x1234, 0x5678, 0x08, 1, 2);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "SS7 TX macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "ss7_msg_tx") != NULL, "Event type correct");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 2 complete\n");
}

// Test 3: SIGTRAN RX macro
static void test_sigtran_rx_macro(void) {
    printf("\n=== Test 3: SIGTRAN RX Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SIGTRAN_RX(0xabcd);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "SIGTRAN RX macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "sigtran_msg_rx") != NULL, "Event type correct");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 3 complete\n");
}

// Test 4: SIGTRAN TX macro
static void test_sigtran_tx_macro(void) {
    printf("\n=== Test 4: SIGTRAN TX Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SIGTRAN_TX(0xabcd);
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "SIGTRAN TX macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "sigtran_msg_tx") != NULL, "Event type correct");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 4 complete\n");
}

// Test 5: Signal anomaly macro
static void test_sig_anomaly_macro(void) {
    printf("\n=== Test 5: Signal Anomaly Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SIG_ANOMALY("ss7", "Unexpected message type");
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "Signal anomaly macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "sig_anomaly") != NULL, "Event type correct");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 5 complete\n");
}

// Test 6: SS7 full macro
static void test_ss7_full_macro(void) {
    printf("\n=== Test 6: SS7 Full Macro ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SS7_FULL(0x1234, 0x5678, 0x08, 1, 2, "STP", "lab");
    
    capture_stderr_stop();
    
    TEST_ASSERT(captured_len > 0, "SS7 full macro logs event");
    TEST_ASSERT(strstr(captured_stderr, "ss7") != NULL, "Stack name in output");
    TEST_ASSERT(strstr(captured_stderr, "STP") != NULL, "Role in output");
    TEST_ASSERT(strstr(captured_stderr, "lab") != NULL, "Environment in output");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 6 complete\n");
}

// Test 7: Multiple macros
static void test_multiple_macros(void) {
    printf("\n=== Test 7: Multiple Macros ===\n");
    
    dsmil_ot_telemetry_init();
    
    capture_stderr_start();
    
    DSMIL_LOG_SS7_RX(0x1111, 0x2222, 0x08, 1, 1);
    DSMIL_LOG_SS7_TX(0x1111, 0x3333, 0x08, 1, 2);
    DSMIL_LOG_SIGTRAN_RX(0xaaaa);
    DSMIL_LOG_SIGTRAN_TX(0xbbbb);
    
    capture_stderr_stop();
    
    int rx_count = 0, tx_count = 0;
    const char *p = captured_stderr;
    while ((p = strstr(p, "ss7_msg_rx")) != NULL) {
        rx_count++;
        p++;
    }
    p = captured_stderr;
    while ((p = strstr(p, "ss7_msg_tx")) != NULL) {
        tx_count++;
        p++;
    }
    
    TEST_ASSERT(rx_count >= 1, "SS7 RX logged");
    TEST_ASSERT(tx_count >= 1, "SS7 TX logged");
    
    dsmil_ot_telemetry_shutdown();
    printf("Test 7 complete\n");
}

int main(void) {
    printf("========================================\n");
    printf("Telecom Macros Test Suite\n");
    printf("========================================\n");
    
    test_ss7_rx_macro();
    test_ss7_tx_macro();
    test_sigtran_rx_macro();
    test_sigtran_tx_macro();
    test_sig_anomaly_macro();
    test_ss7_full_macro();
    test_multiple_macros();
    
    printf("\n========================================\n");
    printf("Test Results: %d passed, %d failed\n", test_passed, test_failed);
    printf("========================================\n");
    
    return test_failed > 0 ? 1 : 0;
}
