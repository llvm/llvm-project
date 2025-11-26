/**
 * @file dsmil_stealth_runtime.c
 * @brief DSLLVM Stealth Mode Runtime Support (v1.4)
 *
 * Runtime support functions for stealth mode transformations.
 * Provides timing, delay, and network batching primitives.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

/**
 * Get current timestamp in nanoseconds
 *
 * @return Timestamp in nanoseconds since epoch
 */
uint64_t dsmil_get_timestamp_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }

    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * Sleep for specified nanoseconds
 *
 * @param ns Nanoseconds to sleep
 *
 * Used for constant-rate execution padding. Ensures functions take
 * predictable time regardless of actual work performed.
 */
void dsmil_nanosleep(uint64_t ns) {
    if (ns == 0)
        return;

    struct timespec req, rem;
    req.tv_sec = ns / 1000000000ULL;
    req.tv_nsec = ns % 1000000000ULL;

    // Handle interrupts by retrying
    while (nanosleep(&req, &rem) == -1) {
        if (errno != EINTR)
            break;
        req = rem;
    }
}

/**
 * Network stealth wrapper for batching/delaying I/O
 *
 * @param data Data to send
 * @param length Length of data
 *
 * Applies timing delays to reduce network fingerprints by normalizing
 * send intervals. This function is called BEFORE the actual network
 * send operation to add controlled delays between transmissions.
 *
 * The pass inserts this wrapper before send/write calls to enforce
 * minimum intervals between network operations, reducing burst patterns
 * and timing-based fingerprinting.
 *
 * IMPORTANT: This wrapper only applies timing delays. The actual network
 * send must still be performed by the original call that follows this
 * wrapper. Do NOT use this as a replacement for send() - it's a timing
 * decorator that runs before the real send.
 */
void dsmil_network_stealth_wrapper(const void *data, uint64_t length) {
    static uint64_t last_send_time = 0;
    uint64_t current_time = dsmil_get_timestamp_ns();

    // Minimum 10ms between sends to reduce burst fingerprinting
    const uint64_t MIN_INTERVAL_NS = 10 * 1000000ULL;

    // Enforce minimum interval between sends
    if (last_send_time != 0) {
        uint64_t elapsed = current_time - last_send_time;
        if (elapsed < MIN_INTERVAL_NS) {
            // Add delay to reach minimum interval
            dsmil_nanosleep(MIN_INTERVAL_NS - elapsed);
        }
    }

    // Update last send timestamp
    last_send_time = dsmil_get_timestamp_ns();

    // Touch parameters to avoid unused warnings
    // The data/length will be used by the actual send() call that follows
    (void)data;
    (void)length;
}

/**
 * Initialize stealth runtime subsystem
 *
 * @return 0 on success, -1 on error
 *
 * Call at program startup to initialize stealth mode resources.
 */
int dsmil_stealth_init(void) {
    // Initialize any global state needed for stealth mode
    // For example, network batching queues, timing calibration, etc.
    return 0;
}

/**
 * Shutdown stealth runtime subsystem
 *
 * Flushes any pending network operations and releases resources.
 */
void dsmil_stealth_shutdown(void) {
    // Flush pending network operations
    // Release any allocated resources
}

/**
 * Get stealth mode status
 *
 * @return 1 if stealth mode active, 0 otherwise
 */
int dsmil_stealth_is_active(void) {
    // Check if runtime is in stealth mode
    // This could be controlled via environment variable or config file
    return 0; // TODO: Implement
}

/**
 * Calibrate constant-rate timing
 *
 * @param target_ms Target execution time in milliseconds
 * @return Calibrated overhead in nanoseconds
 *
 * Measures timing overhead to improve constant-rate accuracy.
 */
uint64_t dsmil_stealth_calibrate_timing(unsigned target_ms) {
    const int ITERATIONS = 100;
    uint64_t total_overhead = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        uint64_t start = dsmil_get_timestamp_ns();
        uint64_t end = dsmil_get_timestamp_ns();
        total_overhead += (end - start);
    }

    return total_overhead / ITERATIONS;
}
