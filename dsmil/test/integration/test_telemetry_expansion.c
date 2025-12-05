/**
 * @file test_telemetry_expansion.c
 * @brief Integration tests for Telemetry Expansion
 *
 * End-to-end tests combining annotations, pass instrumentation, and runtime behavior.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_attributes.h"
#include "dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Test annotated functions
DSMIL_NET_IO
DSMIL_LAYER(4)
void network_connect(const char *host, int port) {
    // Should be instrumented with net_io telemetry
}

DSMIL_CRYPTO
DSMIL_LAYER(3)
int encrypt_data(const uint8_t *plaintext, uint8_t *ciphertext, size_t len) {
    // Should be instrumented with crypto telemetry
    return 0;
}

DSMIL_FILE
DSMIL_LAYER(4)
FILE* open_config_file(const char *filename) {
    // Should be instrumented with file telemetry
    return NULL;
}

DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void handle_error(int code, const char *msg) {
    // Should be instrumented with error telemetry
}

DSMIL_ERROR_HANDLER
DSMIL_LAYER(5)
void panic(const char *msg) {
    // Should be instrumented with panic telemetry (name suggests panic)
    abort();
}

DSMIL_UNTRUSTED
DSMIL_LAYER(7)
void process_user_input(const char *input) {
    // Should be instrumented with untrusted telemetry
}

int main(void) {
    printf("Telemetry Expansion Integration Test\n");
    printf("=====================================\n");
    
    // Initialize telemetry
    dsmil_ot_telemetry_init();
    
    // Test that telemetry level can be queried
    dsmil_telemetry_level_t level = dsmil_telemetry_get_level();
    printf("Telemetry level: %d\n", level);
    assert(level >= DSMIL_TELEMETRY_LEVEL_OFF && level <= DSMIL_TELEMETRY_LEVEL_TRACE);
    
    // Call annotated functions (should trigger instrumentation if compiled with pass)
    network_connect("example.com", 80);
    encrypt_data(NULL, NULL, 0);
    open_config_file("config.txt");
    handle_error(-1, "Test error");
    process_user_input("test input");
    
    // Test level allows function
    int allows_net = dsmil_telemetry_level_allows(DSMIL_TELEMETRY_NET_IO, "net");
    int allows_crypto = dsmil_telemetry_level_allows(DSMIL_TELEMETRY_CRYPTO, "crypto");
    printf("Level allows net: %d\n", allows_net);
    printf("Level allows crypto: %d\n", allows_crypto);
    
    // Test new event types
    dsmil_telemetry_event_t ev_net = {
        .event_type = DSMIL_TELEMETRY_NET_IO,
        .module_id = "test_module",
        .func_id = "network_connect",
        .file = __FILE__,
        .line = __LINE__,
        .category = "net",
        .op = "connect",
        .status_code = 0,
        .resource = "tcp://example.com:80"
    };
    dsmil_telemetry_event(&ev_net);
    
    dsmil_telemetry_event_t ev_crypto = {
        .event_type = DSMIL_TELEMETRY_CRYPTO,
        .module_id = "test_module",
        .func_id = "encrypt_data",
        .file = __FILE__,
        .line = __LINE__,
        .category = "crypto",
        .op = "encrypt",
        .status_code = 0
    };
    dsmil_telemetry_event(&ev_crypto);
    
    dsmil_telemetry_event_t ev_error = {
        .event_type = DSMIL_TELEMETRY_ERROR,
        .module_id = "test_module",
        .func_id = "handle_error",
        .file = __FILE__,
        .line = __LINE__,
        .category = "error",
        .op = "error",
        .status_code = -1,
        .error_msg = "Test error message"
    };
    dsmil_telemetry_event(&ev_error);
    
    dsmil_ot_telemetry_shutdown();
    
    printf("\nIntegration test completed successfully\n");
    return 0;
}
