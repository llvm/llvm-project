/**
 * @file dsmil_telemetry.h
 * @brief DSLLVM Telemetry API (v1.3)
 *
 * Provides telemetry functions for safety-critical and mission-critical
 * code. Integrates with Layer 5 Performance AI and Layer 62 Forensics.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_TELEMETRY_H
#define DSMIL_TELEMETRY_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_TELEMETRY_API Telemetry API
 * @{
 */

/**
 * Telemetry levels (must match mission-profiles.json)
 */
typedef enum {
    DSMIL_TELEMETRY_DISABLED = 0,  /**< No telemetry */
    DSMIL_TELEMETRY_MINIMAL  = 1,  /**< Minimal (border_ops) */
    DSMIL_TELEMETRY_STANDARD = 2,  /**< Standard */
    DSMIL_TELEMETRY_FULL     = 3,  /**< Full (cyber_defence) */
    DSMIL_TELEMETRY_VERBOSE  = 4   /**< Verbose (exercise_only/lab_research) */
} dsmil_telemetry_level_t;

/**
 * Event severity levels
 */
typedef enum {
    DSMIL_EVENT_DEBUG    = 0,  /**< Debug information */
    DSMIL_EVENT_INFO     = 1,  /**< Informational */
    DSMIL_EVENT_WARNING  = 2,  /**< Warning condition */
    DSMIL_EVENT_ERROR    = 3,  /**< Error condition */
    DSMIL_EVENT_CRITICAL = 4   /**< Critical security event */
} dsmil_event_severity_t;

/**
 * Telemetry event structure
 */
typedef struct {
    uint64_t timestamp_ns;           /**< Nanosecond timestamp */
    const char *component;           /**< Component name (crypto, network, etc.) */
    const char *event_name;          /**< Event identifier */
    dsmil_event_severity_t severity; /**< Event severity */
    uint32_t layer;                  /**< DSMIL layer (0-8) */
    uint32_t device;                 /**< DSMIL device (0-103) */
    const char *message;             /**< Optional message */
    uint64_t metadata[4];            /**< Optional metadata */
} dsmil_event_t;

/**
 * Telemetry configuration
 */
typedef struct {
    dsmil_telemetry_level_t level;   /**< Current telemetry level */
    const char *mission_profile;     /**< Active mission profile */
    int (*sink_fn)(const dsmil_event_t *event);  /**< Event sink callback */
    void *sink_context;              /**< Sink context pointer */
} dsmil_telemetry_config_t;

/**
 * @name Core Telemetry Functions
 * @{
 */

/**
 * Initialize telemetry subsystem
 *
 * @param config Telemetry configuration
 * @return 0 on success, negative on error
 *
 * Must be called before any telemetry functions. Typically called
 * during process initialization based on mission profile.
 *
 * Example:
 * @code
 * dsmil_telemetry_config_t config = {
 *     .level = DSMIL_TELEMETRY_FULL,
 *     .mission_profile = "cyber_defence",
 *     .sink_fn = my_event_sink,
 *     .sink_context = NULL
 * };
 * dsmil_telemetry_init(&config);
 * @endcode
 */
int dsmil_telemetry_init(const dsmil_telemetry_config_t *config);

/**
 * Shutdown telemetry subsystem
 *
 * Flushes any pending events and releases resources.
 */
void dsmil_telemetry_shutdown(void);

/**
 * Get current telemetry level
 *
 * @return Current telemetry level
 */
dsmil_telemetry_level_t dsmil_telemetry_get_level(void);

/**
 * Set telemetry level at runtime
 *
 * @param level New telemetry level
 *
 * Note: Some mission profiles may prevent runtime level changes
 */
void dsmil_telemetry_set_level(dsmil_telemetry_level_t level);

/** @} */

/**
 * @name Counter Telemetry
 * @{
 */

/**
 * Increment a named counter
 *
 * @param counter_name Counter identifier (e.g., "ml_kem_calls")
 *
 * Atomically increments a monotonic counter. Counters are used for:
 * - Call frequency analysis (Layer 5 Performance AI)
 * - Usage statistics
 * - Rate limiting decisions
 *
 * Example:
 * @code
 * DSMIL_SAFETY_CRITICAL("crypto")
 * void ml_kem_encapsulate(...) {
 *     dsmil_counter_inc("ml_kem_encapsulate_calls");
 *     // ... operation ...
 * }
 * @endcode
 *
 * @note Thread-safe
 * @note Zero overhead if telemetry level is DISABLED
 */
void dsmil_counter_inc(const char *counter_name);

/**
 * Add value to a named counter
 *
 * @param counter_name Counter identifier
 * @param value Value to add
 *
 * Example:
 * @code
 * void process_batch(size_t count) {
 *     dsmil_counter_add("items_processed", count);
 * }
 * @endcode
 */
void dsmil_counter_add(const char *counter_name, uint64_t value);

/**
 * Get current counter value
 *
 * @param counter_name Counter identifier
 * @return Current counter value
 */
uint64_t dsmil_counter_get(const char *counter_name);

/**
 * Reset counter to zero
 *
 * @param counter_name Counter identifier
 */
void dsmil_counter_reset(const char *counter_name);

/** @} */

/**
 * @name Event Telemetry
 * @{
 */

/**
 * Log a telemetry event
 *
 * @param event_name Event identifier
 *
 * Simple event logging with INFO severity.
 *
 * Example:
 * @code
 * DSMIL_MISSION_CRITICAL
 * int detect_threat(...) {
 *     dsmil_event_log("threat_detection_start");
 *     // ... detection logic ...
 *     dsmil_event_log("threat_detection_complete");
 * }
 * @endcode
 */
void dsmil_event_log(const char *event_name);

/**
 * Log event with severity
 *
 * @param event_name Event identifier
 * @param severity Event severity level
 *
 * Example:
 * @code
 * if (validation_failed) {
 *     dsmil_event_log_severity("input_validation_failed", DSMIL_EVENT_ERROR);
 * }
 * @endcode
 */
void dsmil_event_log_severity(const char *event_name, dsmil_event_severity_t severity);

/**
 * Log event with message
 *
 * @param event_name Event identifier
 * @param severity Event severity level
 * @param message Human-readable message
 *
 * Example:
 * @code
 * dsmil_event_log_msg("crypto_error", DSMIL_EVENT_ERROR,
 *                     "ML-KEM decapsulation failed");
 * @endcode
 */
void dsmil_event_log_msg(const char *event_name,
                         dsmil_event_severity_t severity,
                         const char *message);

/**
 * Log structured event
 *
 * @param event Full event structure with metadata
 *
 * Most flexible event logging for complex scenarios.
 *
 * Example:
 * @code
 * dsmil_event_t event = {
 *     .timestamp_ns = get_timestamp_ns(),
 *     .component = "network",
 *     .event_name = "packet_received",
 *     .severity = DSMIL_EVENT_INFO,
 *     .layer = 8,
 *     .device = 80,
 *     .message = "High-risk packet detected",
 *     .metadata = {packet_size, source_ip, dest_port, threat_score}
 * };
 * dsmil_event_log_structured(&event);
 * @endcode
 */
void dsmil_event_log_structured(const dsmil_event_t *event);

/** @} */

/**
 * @name Performance Metrics
 * @{
 */

/**
 * Start timing operation
 *
 * @param operation_name Operation identifier
 * @return Timing handle (opaque)
 *
 * Used with dsmil_perf_end() for performance measurement.
 *
 * Example:
 * @code
 * void *timer = dsmil_perf_start("inference_latency");
 * run_inference();
 * dsmil_perf_end(timer);
 * @endcode
 */
void *dsmil_perf_start(const char *operation_name);

/**
 * End timing operation and record duration
 *
 * @param handle Timing handle from dsmil_perf_start()
 *
 * Records duration in microseconds and sends to Layer 5 Performance AI.
 */
void dsmil_perf_end(void *handle);

/**
 * Record latency measurement
 *
 * @param operation_name Operation identifier
 * @param latency_us Latency in microseconds
 *
 * Direct latency recording without start/end pairing.
 */
void dsmil_perf_latency(const char *operation_name, uint64_t latency_us);

/**
 * Record throughput measurement
 *
 * @param operation_name Operation identifier
 * @param items_per_sec Items processed per second
 */
void dsmil_perf_throughput(const char *operation_name, double items_per_sec);

/** @} */

/**
 * @name Layer 62 Forensics Integration
 * @{
 */

/**
 * Create forensic checkpoint
 *
 * @param checkpoint_name Checkpoint identifier
 *
 * Creates a forensic snapshot for post-incident analysis.
 * Captures:
 * - Current call stack
 * - Active counters
 * - Recent events
 * - Memory allocations
 *
 * Example:
 * @code
 * DSMIL_MISSION_CRITICAL
 * int execute_sensitive_operation() {
 *     dsmil_forensic_checkpoint("pre_operation");
 *     int result = do_operation();
 *     dsmil_forensic_checkpoint("post_operation");
 *     return result;
 * }
 * @endcode
 */
void dsmil_forensic_checkpoint(const char *checkpoint_name);

/**
 * Log security event for forensics
 *
 * @param event_name Event identifier
 * @param severity Event severity
 * @param details Additional details (JSON string or NULL)
 *
 * Security-relevant events that may be used in incident response.
 */
void dsmil_forensic_security_event(const char *event_name,
                                   dsmil_event_severity_t severity,
                                   const char *details);

/** @} */

/**
 * @name Mission Profile Integration
 * @{
 */

/**
 * Check if telemetry is required by mission profile
 *
 * @return 1 if telemetry required, 0 otherwise
 *
 * Query at runtime if current mission profile requires telemetry.
 */
int dsmil_telemetry_is_required(void);

/**
 * Validate function has telemetry
 *
 * @param function_name Function name to check
 * @return 1 if function has telemetry calls, 0 otherwise
 *
 * Runtime validation for dynamic scenarios.
 */
int dsmil_telemetry_validate_function(const char *function_name);

/** @} */

/**
 * @name Telemetry Sinks
 * @{
 */

/**
 * Register custom telemetry sink
 *
 * @param sink_fn Event sink callback
 * @param context Opaque context pointer
 * @return 0 on success, negative on error
 *
 * Custom sinks can export telemetry to:
 * - Prometheus/OpenMetrics
 * - StatsD
 * - Layer 5 Performance AI service
 * - Layer 62 Forensics database
 * - Custom logging systems
 *
 * Example:
 * @code
 * int my_sink(const dsmil_event_t *event) {
 *     fprintf(stderr, "[%s] %s: %s\n",
 *             event->component, event->event_name, event->message);
 *     return 0;
 * }
 *
 * dsmil_telemetry_register_sink(my_sink, NULL);
 * @endcode
 */
int dsmil_telemetry_register_sink(
    int (*sink_fn)(const dsmil_event_t *event),
    void *context);

/**
 * Built-in sink: stdout logging
 */
int dsmil_telemetry_sink_stdout(const dsmil_event_t *event);

/**
 * Built-in sink: syslog
 */
int dsmil_telemetry_sink_syslog(const dsmil_event_t *event);

/**
 * Built-in sink: Prometheus exporter
 */
int dsmil_telemetry_sink_prometheus(const dsmil_event_t *event);

/** @} */

/** @} */  // End of DSMIL_TELEMETRY_API

#ifdef __cplusplus
}
#endif

#endif // DSMIL_TELEMETRY_H
