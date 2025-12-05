/**
 * @file dsssl_fuzz_example.c
 * @brief Example DSSSL code with fuzzing instrumentation
 *
 * Demonstrates how to annotate DSSSL code for fuzzing and telemetry.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsssl_fuzz_attributes.h"
#include "dsssl_fuzz_telemetry.h"
#include <stdint.h>
#include <stddef.h>

/**
 * TLS handshake processing function
 * 
 * Marked with state machine annotation for transition tracking.
 */
DSSSL_STATE_MACHINE("tls_handshake")
DSSSL_COVERAGE
int tls_process_handshake(void *ssl, const uint8_t *data, size_t len) {
    if (len < 1) {
        return -1;
    }

    uint8_t msg_type = data[0];
    
    // State transitions would be tracked automatically
    switch (msg_type) {
    case 1:  // ClientHello
        dsssl_state_transition(1, 0, 1);  // SM ID 1: TLS handshake
        break;
    case 2:  // ServerHello
        dsssl_state_transition(1, 1, 2);
        break;
    default:
        return -1;
    }

    return 0;
}

/**
 * ECDSA signing function
 * 
 * Marked with crypto annotation for metric collection.
 */
DSSSL_CRYPTO("ecdsa_sign")
DSSSL_COVERAGE
int ecdsa_sign(const void *key, uint8_t *sig, size_t *sig_len,
               const uint8_t *msg, size_t msg_len) {
    dsssl_crypto_metric_begin("ecdsa_sign");
    
    // Simulate signing operation
    // In real code, this would call actual ECDSA implementation
    
    uint32_t branches = 0;
    uint32_t loads = 0;
    uint32_t stores = 0;
    
    // Count operations (simplified - real implementation would track dynamically)
    for (size_t i = 0; i < msg_len; i++) {
        loads++;
        if (msg[i] > 128) {
            branches++;
        }
    }
    stores = *sig_len;
    
    dsssl_crypto_metric_record("ecdsa_sign", branches, loads, stores, 0);
    dsssl_crypto_metric_end("ecdsa_sign");
    
    // Check budget
    if (dsssl_crypto_check_budget("ecdsa_sign", branches, loads, stores, 0)) {
        // Budget violated - would trigger telemetry event
    }
    
    return 0;
}

/**
 * AEAD initialization with misuse detection
 */
DSSSL_API_MISUSE_CHECK("AEAD_init")
DSSSL_COVERAGE
int aead_init(void *ctx, const void *aead, const uint8_t *key, size_t key_len,
              const uint8_t *nonce, size_t nonce_len) {
    // Check nonce length
    if (nonce_len < 12) {
        dsssl_api_misuse_report("AEAD_init", "nonce_too_short", 
                               dsssl_fuzz_get_context());
        return -1;
    }
    
    // Check for nonce reuse (simplified check)
    static uint8_t last_nonce[16] = {0};
    int nonce_reused = 1;
    for (size_t i = 0; i < nonce_len && i < 16; i++) {
        if (nonce[i] != last_nonce[i]) {
            nonce_reused = 0;
            break;
        }
    }
    
    if (nonce_reused) {
        dsssl_api_misuse_report("AEAD_init", "nonce_reuse",
                               dsssl_fuzz_get_context());
    }
    
    // Copy nonce for next check
    for (size_t i = 0; i < nonce_len && i < 16; i++) {
        last_nonce[i] = nonce[i];
    }
    
    return 0;
}

/**
 * Session ticket handling
 */
DSSSL_STATE_MACHINE("ticket_lifecycle")
DSSSL_COVERAGE
int handle_session_ticket(uint64_t ticket_id, int action) {
    switch (action) {
    case 1:  // Issue
        dsssl_ticket_event(DSSSL_TICKET_ISSUE, ticket_id);
        dsssl_state_transition(2, 0, 1);  // SM ID 2: ticket lifecycle
        break;
    case 2:  // Use
        dsssl_ticket_event(DSSSL_TICKET_USE, ticket_id);
        dsssl_state_transition(2, 1, 2);
        break;
    case 3:  // Expire
        dsssl_ticket_event(DSSSL_TICKET_EXPIRE, ticket_id);
        dsssl_state_transition(2, 2, 0);
        break;
    }
    
    return 0;
}

/**
 * Example usage
 */
int main(void) {
    // Initialize telemetry
    dsssl_fuzz_telemetry_init("dsssl_fuzz_telemetry.yaml", 65536);
    
    // Set context ID (would be hash of fuzz input in real harness)
    dsssl_fuzz_set_context(0x1234567890ABCDEF);
    
    // Simulate operations
    uint8_t handshake_data[] = {1, 2, 3, 4, 5};
    tls_process_handshake(NULL, handshake_data, sizeof(handshake_data));
    
    uint8_t sig[64];
    size_t sig_len = 64;
    uint8_t msg[] = "Hello, World!";
    ecdsa_sign(NULL, sig, &sig_len, msg, sizeof(msg) - 1);
    
    uint8_t nonce[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    aead_init(NULL, NULL, NULL, 0, nonce, 12);
    
    handle_session_ticket(0xABCDEF, 1);
    
    // Flush telemetry
    dsssl_fuzz_flush_events("telemetry.bin");
    
    dsssl_fuzz_telemetry_shutdown();
    return 0;
}
