/**
 * @file generic_fuzz_example.c
 * @brief Example Generic Fuzzing Target with DSLLVM Instrumentation
 *
 * Demonstrates how to annotate any code for fuzzing and telemetry.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_fuzz_attributes.h"
#include "dsmil_fuzz_telemetry.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/**
 * Example: HTTP parser state machine
 */
DSMIL_FUZZ_STATE_MACHINE("http_parser")
DSMIL_FUZZ_COVERAGE
DSMIL_FUZZ_ENTRY_POINT
int http_parse_request(const uint8_t *data, size_t len) {
    if (len < 4) {
        return -1;
    }

    // Track state transitions
    uint8_t state = 0;  // START
    
    for (size_t i = 0; i < len; i++) {
        if (data[i] == 'G' && state == 0) {
            dsmil_fuzz_state_transition(1, 0, 1);  // SM ID 1: HTTP parser
            state = 1;  // METHOD
        } else if (data[i] == ' ' && state == 1) {
            dsmil_fuzz_state_transition(1, 1, 2);  // URI
            state = 2;
        } else if (data[i] == '\r' && state == 2) {
            dsmil_fuzz_state_transition(1, 2, 3);  // HEADERS
            state = 3;
        }
    }
    
    return 0;
}

/**
 * Example: JSON parser (critical operation)
 */
DSMIL_FUZZ_CRITICAL_OP("json_parse")
DSMIL_FUZZ_COVERAGE
int json_parse(const char *json_str, size_t len) {
    dsmil_fuzz_metric_begin("json_parse");
    
    if (!json_str || len == 0) {
        return -1;
    }
    
    uint32_t branches = 0;
    uint32_t loads = 0;
    uint32_t stores = 0;
    
    // Simulate parsing
    for (size_t i = 0; i < len; i++) {
        loads++;
        if (json_str[i] == '{') {
            branches++;
        } else if (json_str[i] == '}') {
            branches++;
        } else if (json_str[i] == '"') {
            branches++;
        }
    }
    
    dsmil_fuzz_metric_record("json_parse", branches, loads, stores, 0);
    dsmil_fuzz_metric_end("json_parse");
    
    // Check budget
    if (dsmil_fuzz_check_budget("json_parse", branches, loads, stores, 0)) {
        // Budget violated
    }
    
    return 0;
}

/**
 * Example: Buffer operation with misuse detection
 */
DSMIL_FUZZ_API_MISUSE_CHECK("buffer_write")
DSMIL_FUZZ_COVERAGE
int buffer_write(void *buf, size_t buf_size, const void *data, size_t data_len) {
    // Check bounds
    if (data_len > buf_size) {
        dsmil_fuzz_api_misuse_report("buffer_write", "buffer_overflow",
                                     dsmil_fuzz_get_context());
        return -1;
    }
    
    // Check null pointer
    if (!buf || !data) {
        dsmil_fuzz_api_misuse_report("buffer_write", "null_pointer",
                                     dsmil_fuzz_get_context());
        return -1;
    }
    
    memcpy(buf, data, data_len);
    return 0;
}

/**
 * Example: Stateful operation
 */
DSMIL_FUZZ_STATE_MACHINE("session_manager")
DSMIL_FUZZ_COVERAGE
int create_session(uint64_t session_id) {
    dsmil_fuzz_state_event(DSMIL_STATE_CREATE, session_id);
    dsmil_fuzz_state_transition(2, 0, 1);  // SM ID 2: session manager
    return 0;
}

int use_session(uint64_t session_id) {
    dsmil_fuzz_state_event(DSMIL_STATE_USE, session_id);
    dsmil_fuzz_state_transition(2, 1, 2);
    return 0;
}

int destroy_session(uint64_t session_id) {
    dsmil_fuzz_state_event(DSMIL_STATE_DESTROY, session_id);
    dsmil_fuzz_state_transition(2, 2, 0);
    return 0;
}

/**
 * Example usage
 */
int main(void) {
    // Initialize telemetry
    dsmil_fuzz_telemetry_init("fuzz_telemetry_generic.yaml", 65536);
    
    // Set context ID
    dsmil_fuzz_set_context(0x1234567890ABCDEF);
    
    // Example inputs
    const char *http_req = "GET /index.html HTTP/1.1\r\n";
    http_parse_request((const uint8_t*)http_req, strlen(http_req));
    
    const char *json = "{\"key\":\"value\"}";
    json_parse(json, strlen(json));
    
    char buf[100];
    buffer_write(buf, sizeof(buf), "test", 4);
    
    create_session(0xABCD);
    use_session(0xABCD);
    destroy_session(0xABCD);
    
    // Flush telemetry
    dsmil_fuzz_flush_events("telemetry.bin");
    
    dsmil_fuzz_telemetry_shutdown();
    return 0;
}
