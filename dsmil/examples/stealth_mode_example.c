/**
 * @file stealth_mode_example.c
 * @brief DSLLVM Stealth Mode Example (Feature 2.1)
 *
 * Demonstrates stealth mode attributes and transformations for
 * low-signature execution in hostile network environments.
 *
 * Compile:
 *   dsmil-clang -fdsmil-mission-profile=covert_ops \
 *               -O3 -o stealth_example stealth_mode_example.c
 *
 * Or with explicit stealth flags:
 *   dsmil-clang -dsmil-stealth-mode=aggressive \
 *               -dsmil-stealth-strip-telemetry \
 *               -dsmil-stealth-constant-rate \
 *               -O3 -o stealth_example stealth_mode_example.c
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <dsmil_attributes.h>
#include <dsmil_telemetry.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Example 1: Basic stealth function
 *
 * This function uses the simple DSMIL_STEALTH attribute to enable
 * standard stealth transformations.
 */
DSMIL_STEALTH
DSMIL_LAYER(7)
void stealth_data_processing(const uint8_t *data, size_t len) {
    // This telemetry will be stripped in stealth mode
    dsmil_counter_inc("data_processing_calls");

    // Process data
    for (size_t i = 0; i < len; i++) {
        // Actual processing would happen here
        (void)data[i];
    }

    // This verbose logging will also be stripped
    dsmil_event_log("data_processing_complete");
}

/**
 * Example 2: Aggressive stealth with constant-rate execution
 *
 * This function uses aggressive stealth mode with constant-rate
 * execution to prevent timing pattern analysis.
 */
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_CONSTANT_RATE
DSMIL_LAYER(7)
void constant_rate_heartbeat(void) {
    // This function will always take exactly the target time
    // (default 100ms) regardless of work performed

    // Critical telemetry is preserved even in aggressive mode
    // if function is marked safety_critical
    dsmil_counter_inc("heartbeat_calls");

    // Do actual work
    // ... network check, status update, etc. ...

    // Compiler will add timing padding to ensure constant execution time
}

/**
 * Example 3: Network stealth for covert communication
 *
 * This function combines low-signature mode with network stealth
 * to reduce fingerprints.
 */
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_NETWORK_STEALTH
DSMIL_LAYER(7)
void covert_status_update(const char *status_msg) {
    // Network I/O will be batched and delayed to reduce patterns
    // send_network_packet(status_msg);

    // Minimal telemetry
    dsmil_counter_inc("status_updates");

    // Verbose telemetry stripped
    // dsmil_event_log("status_update_sent"); // This will be removed
}

/**
 * Example 4: Safety-critical function with stealth
 *
 * Even in stealth mode, safety-critical functions retain
 * minimum required telemetry.
 */
DSMIL_SAFETY_CRITICAL("crypto")
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_SECRET
DSMIL_LAYER(8)
void crypto_operation(const uint8_t *key, const uint8_t *data, uint8_t *output) {
    // This critical telemetry is ALWAYS preserved
    dsmil_counter_inc("crypto_operations");

    // Constant-time crypto operations
    for (int i = 0; i < 32; i++) {
        output[i] = key[i] ^ data[i];
    }

    // Critical security event - always logged
    dsmil_forensic_security_event("crypto_op_complete",
                                  DSMIL_EVENT_INFO,
                                  NULL);
}

/**
 * Example 5: Jitter suppression for predictable timing
 *
 * This function uses jitter suppression to minimize timing variance.
 */
DSMIL_LOW_SIGNATURE("standard")
DSMIL_JITTER_SUPPRESS
DSMIL_LAYER(7)
void predictable_timing_operation(void) {
    // Function will have minimal timing variance
    // - No dynamic frequency scaling
    // - Consistent cache behavior
    // - Predictable execution time

    // Do work with predictable timing
    for (int i = 0; i < 1000; i++) {
        // Work here
    }
}

/**
 * Example 6: Covert ops main entry point
 *
 * Demonstrates full stealth configuration for covert operations.
 */
DSMIL_MISSION_PROFILE("covert_ops")
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_SANDBOX("l7_covert_ops")
int main(int argc, char **argv) {
    printf("DSLLVM Stealth Mode Example\n");
    printf("Mission Profile: covert_ops\n");
    printf("Stealth Level: aggressive\n\n");

    // Initialize stealth runtime
    // dsmil_stealth_init();

    // Example data
    uint8_t data[] = {0x01, 0x02, 0x03, 0x04};

    // Example 1: Basic stealth processing
    stealth_data_processing(data, sizeof(data));

    // Example 2: Constant-rate heartbeat
    constant_rate_heartbeat();

    // Example 3: Covert network update
    covert_status_update("System operational");

    // Example 4: Safety-critical crypto
    uint8_t key[32] = {0};
    uint8_t output[32] = {0};
    crypto_operation(key, data, output);

    // Example 5: Predictable timing
    predictable_timing_operation();

    printf("All stealth operations complete\n");

    // Cleanup
    // dsmil_stealth_shutdown();

    return 0;
}

/**
 * Example 7: Comparison - Normal vs Stealth
 *
 * This example shows the difference between normal and stealth modes.
 */

// Normal mode - full telemetry
DSMIL_LAYER(7)
void normal_function(void) {
    dsmil_counter_inc("normal_calls");
    dsmil_event_log("normal_start");

    // Do work
    for (int i = 0; i < 100; i++) {
        // Work here
    }

    dsmil_perf_latency("normal_function", 50);
    dsmil_event_log("normal_complete");
}

// Stealth mode - minimal telemetry
DSMIL_STEALTH
DSMIL_LAYER(7)
void stealth_function(void) {
    dsmil_counter_inc("stealth_calls");
    dsmil_event_log("stealth_start"); // Will be stripped

    // Do work (same as normal)
    for (int i = 0; i < 100; i++) {
        // Work here
    }

    dsmil_perf_latency("stealth_function", 50); // Will be stripped
    dsmil_event_log("stealth_complete"); // Will be stripped
}

/**
 * Stealth Mode Summary
 *
 * Transformations Applied:
 *
 * STEALTH_MINIMAL:
 *   - Strip verbose/debug telemetry
 *   - Keep critical and standard telemetry
 *   - No timing transformations
 *
 * STEALTH_STANDARD:
 *   - Strip verbose and performance telemetry
 *   - Keep critical telemetry only
 *   - Jitter suppression enabled
 *   - Network fingerprint reduction
 *
 * STEALTH_AGGRESSIVE:
 *   - Strip all non-critical telemetry
 *   - Constant-rate execution
 *   - Maximum jitter suppression
 *   - Aggressive network batching
 *   - Minimal forensic signature
 *
 * Trade-offs:
 *   + Reduced detectability
 *   + Lower network fingerprint
 *   + Harder to analyze via timing
 *   - Reduced observability
 *   - Harder to debug issues
 *   - Potential performance impact
 *
 * Best Practices:
 *   1. Use covert_ops or border_ops_stealth mission profiles
 *   2. Mark safety-critical functions to preserve minimum telemetry
 *   3. Maintain high-fidelity test builds for debugging
 *   4. Combine with post-mission data exfiltration
 *   5. Let Layer 5/8 AI model detectability trade-offs
 */
