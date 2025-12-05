/**
 * @file dsmil_fuzz_telemetry.h
 * @brief DSLLVM General-Purpose Fuzzing & Telemetry Runtime API
 *
 * Provides runtime APIs for coverage tracking, state machine instrumentation,
 * metrics collection, and API misuse detection in fuzzing builds.
 * General-purpose foundation for any fuzzing target.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_FUZZ_TELEMETRY_H
#define DSMIL_FUZZ_TELEMETRY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_FUZZ_TELEMETRY_API General Fuzzing & Telemetry API
 * @{
 */

/**
 * Telemetry event types
 */
typedef enum {
    DSMIL_FUZZ_EVENT_COVERAGE_HIT = 1,      /**< Coverage site hit */
    DSMIL_FUZZ_EVENT_STATE_TRANSITION = 2,  /**< State machine transition */
    DSMIL_FUZZ_EVENT_METRIC = 3,            /**< Operation metric */
    DSMIL_FUZZ_EVENT_API_MISUSE = 4,        /**< API misuse detected */
    DSMIL_FUZZ_EVENT_DECISION = 5,          /**< Decision point */
    DSMIL_FUZZ_EVENT_STATE_EVENT = 6,       /**< State event */
    DSMIL_FUZZ_EVENT_BUDGET_VIOLATION = 7   /**< Budget violation */
} dsmil_fuzz_event_type_t;

/**
 * State event subtypes
 */
typedef enum {
    DSMIL_STATE_CREATE = 1,    /**< State created */
    DSMIL_STATE_USE = 2,       /**< State used */
    DSMIL_STATE_DESTROY = 3,   /**< State destroyed */
    DSMIL_STATE_REJECT = 4     /**< State rejected */
} dsmil_state_event_t;

/**
 * Telemetry event structure
 */
typedef struct {
    dsmil_fuzz_event_type_t event_type;    /**< Event type */
    uint64_t timestamp;                    /**< Timestamp (nanoseconds) */
    uint32_t thread_id;                    /**< Thread ID */
    uint64_t context_id;                   /**< Context ID (fuzz input hash) */
    
    union {
        struct {
            uint32_t site_id;              /**< Coverage site ID */
        } coverage;
        
        struct {
            uint16_t sm_id;                /**< State machine ID */
            uint16_t state_from;           /**< Source state */
            uint16_t state_to;             /**< Destination state */
        } state_transition;
        
        struct {
            const char *op_name;           /**< Operation name */
            uint32_t branches;             /**< Branch count */
            uint32_t loads;                /**< Load count */
            uint32_t stores;               /**< Store count */
            uint64_t cycles;               /**< Cycle count (if enabled) */
        } metric;
        
        struct {
            const char *api;               /**< API name */
            const char *reason;            /**< Misuse reason */
            uint64_t context_id;           /**< Context ID */
        } api_misuse;
        
        struct {
            const char *decision;          /**< Decision (accept/reject) */
            uint32_t depth;                /**< Decision depth */
        } decision;
        
        struct {
            dsmil_state_event_t subtype;  /**< State event subtype */
            uint64_t state_id;             /**< State identifier */
        } state_event;
        
        struct {
            const char *budget_name;       /**< Budget name */
            uint64_t actual;               /**< Actual value */
            uint64_t limit;                /**< Limit value */
        } budget_violation;
    } data;
} dsmil_fuzz_telemetry_event_t;

/**
 * Initialize fuzzing telemetry subsystem
 *
 * @param config_path Path to YAML config file (can be NULL)
 * @param ring_buffer_size Size of ring buffer for events (0 = use default)
 * @return 0 on success, negative on error
 */
int dsmil_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size);

/**
 * Shutdown telemetry subsystem
 */
void dsmil_fuzz_telemetry_shutdown(void);

/**
 * Set context ID for current fuzz input
 *
 * @param context_id Context ID (typically hash of fuzz input)
 */
void dsmil_fuzz_set_context(uint64_t context_id);

/**
 * Get current context ID
 */
uint64_t dsmil_fuzz_get_context(void);

/**
 * @name Coverage Instrumentation
 * @{
 */

/**
 * Record coverage site hit
 *
 * @param site_id Coverage site ID (assigned by instrumentation pass)
 */
void dsmil_fuzz_cov_hit(uint32_t site_id);

/** @} */

/**
 * @name State Machine Instrumentation
 * @{
 */

/**
 * Record state machine transition
 *
 * @param sm_id State machine ID
 * @param state_from Source state
 * @param state_to Destination state
 */
void dsmil_fuzz_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to);

/** @} */

/**
 * @name Metrics Instrumentation
 * @{
 */

/**
 * Begin operation metric collection
 *
 * @param op_name Operation name
 */
void dsmil_fuzz_metric_begin(const char *op_name);

/**
 * End operation metric collection
 *
 * @param op_name Operation name (must match begin)
 */
void dsmil_fuzz_metric_end(const char *op_name);

/**
 * Record metric values
 *
 * @param op_name Operation name
 * @param branches Branch count
 * @param loads Load count
 * @param stores Store count
 * @param cycles Cycle count (0 if not measured)
 */
void dsmil_fuzz_metric_record(const char *op_name, uint32_t branches,
                              uint32_t loads, uint32_t stores, uint64_t cycles);

/** @} */

/**
 * @name API Misuse Detection
 * @{
 */

/**
 * Report API misuse
 *
 * @param api API name
 * @param reason Misuse reason
 * @param context_id Context ID
 */
void dsmil_fuzz_api_misuse_report(const char *api, const char *reason, uint64_t context_id);

/** @} */

/**
 * @name State Events
 * @{
 */

/**
 * Record state event
 *
 * @param subtype State event subtype
 * @param state_id State identifier
 */
void dsmil_fuzz_state_event(dsmil_state_event_t subtype, uint64_t state_id);

/** @} */

/**
 * @name Telemetry Export
 * @{
 */

/**
 * Get telemetry events from ring buffer
 *
 * @param events Output buffer for events
 * @param max_events Maximum number of events to retrieve
 * @return Number of events retrieved
 */
size_t dsmil_fuzz_get_events(dsmil_fuzz_telemetry_event_t *events, size_t max_events);

/**
 * Flush telemetry events to file
 *
 * @param filepath Output file path
 * @return 0 on success, negative on error
 */
int dsmil_fuzz_flush_events(const char *filepath);

/**
 * Clear telemetry ring buffer
 */
void dsmil_fuzz_clear_events(void);

/** @} */

/**
 * @name Budget Enforcement
 * @{
 */

/**
 * Check operation budget
 *
 * @param op_name Operation name
 * @param branches Branch count
 * @param loads Load count
 * @param stores Store count
 * @param cycles Cycle count
 * @return 0 if within budget, 1 if violated
 */
int dsmil_fuzz_check_budget(const char *op_name, uint32_t branches,
                            uint32_t loads, uint32_t stores, uint64_t cycles);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_FUZZ_TELEMETRY_H */
