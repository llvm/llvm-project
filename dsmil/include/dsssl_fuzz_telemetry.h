/**
 * @file dsssl_fuzz_telemetry.h
 * @brief DSSSL Fuzzing & Telemetry Runtime API
 *
 * Provides runtime APIs for coverage tracking, state machine instrumentation,
 * crypto metrics, and API misuse detection in DSSSL fuzzing builds.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSSSL_FUZZ_TELEMETRY_H
#define DSSSL_FUZZ_TELEMETRY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSSSL_FUZZ_TELEMETRY_API Fuzzing & Telemetry API
 * @{
 */

/**
 * Telemetry event types
 */
typedef enum {
    DSSSL_EVENT_COVERAGE_HIT = 1,      /**< Coverage site hit */
    DSSSL_EVENT_STATE_TRANSITION = 2,  /**< State machine transition */
    DSSSL_EVENT_CRYPTO_METRIC = 3,     /**< Crypto operation metric */
    DSSSL_EVENT_API_MISUSE = 4,        /**< API misuse detected */
    DSSSL_EVENT_PKI_DECISION = 5,      /**< PKI validation decision */
    DSSSL_EVENT_TICKET_EVENT = 6,      /**< Ticket issue/use/expire */
    DSSSL_EVENT_BUDGET_VIOLATION = 7   /**< Budget violation */
} dsssl_event_type_t;

/**
 * Ticket event subtypes
 */
typedef enum {
    DSSSL_TICKET_ISSUE = 1,    /**< Ticket issued */
    DSSSL_TICKET_USE = 2,      /**< Ticket used */
    DSSSL_TICKET_EXPIRE = 3,   /**< Ticket expired */
    DSSSL_TICKET_REJECT = 4    /**< Ticket rejected */
} dsssl_ticket_event_t;

/**
 * Telemetry event structure
 */
typedef struct {
    dsssl_event_type_t event_type;    /**< Event type */
    uint64_t timestamp;                /**< Timestamp (nanoseconds) */
    uint32_t thread_id;                /**< Thread ID */
    uint64_t context_id;               /**< Context ID (fuzz input hash) */
    
    union {
        struct {
            uint32_t site_id;          /**< Coverage site ID */
        } coverage;
        
        struct {
            uint16_t sm_id;            /**< State machine ID */
            uint16_t state_from;       /**< Source state */
            uint16_t state_to;         /**< Destination state */
        } state_transition;
        
        struct {
            const char *op_name;       /**< Operation name */
            uint32_t branches;         /**< Branch count */
            uint32_t loads;            /**< Load count */
            uint32_t stores;           /**< Store count */
            uint64_t cycles;           /**< Cycle count (if enabled) */
        } crypto_metric;
        
        struct {
            const char *api;           /**< API name */
            const char *reason;        /**< Misuse reason */
            uint64_t context_id;       /**< Context ID */
        } api_misuse;
        
        struct {
            const char *decision;      /**< Decision (accept/reject) */
            uint32_t chain_len;        /**< Certificate chain length */
        } pki_decision;
        
        struct {
            dsssl_ticket_event_t subtype; /**< Ticket event subtype */
            uint64_t ticket_id;        /**< Ticket identifier */
        } ticket_event;
        
        struct {
            const char *budget_name;   /**< Budget name */
            uint64_t actual;           /**< Actual value */
            uint64_t limit;            /**< Limit value */
        } budget_violation;
    } data;
} dsssl_telemetry_event_t;

/**
 * Initialize fuzzing telemetry subsystem
 *
 * @param config_path Path to YAML config file (can be NULL)
 * @param ring_buffer_size Size of ring buffer for events (0 = use default)
 * @return 0 on success, negative on error
 */
int dsssl_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size);

/**
 * Shutdown telemetry subsystem
 *
 * Flushes pending events and releases resources.
 */
void dsssl_fuzz_telemetry_shutdown(void);

/**
 * Set context ID for current fuzz input
 *
 * @param context_id Context ID (typically hash of fuzz input)
 */
void dsssl_fuzz_set_context(uint64_t context_id);

/**
 * Get current context ID
 *
 * @return Current context ID
 */
uint64_t dsssl_fuzz_get_context(void);

/**
 * @name Coverage Instrumentation
 * @{
 */

/**
 * Record coverage site hit
 *
 * @param site_id Coverage site ID (assigned by instrumentation pass)
 */
void dsssl_cov_hit(uint32_t site_id);

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
void dsssl_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to);

/** @} */

/**
 * @name Crypto Metrics Instrumentation
 * @{
 */

/**
 * Begin crypto operation metric collection
 *
 * @param op_name Operation name (e.g., "ecdsa_sign")
 */
void dsssl_crypto_metric_begin(const char *op_name);

/**
 * End crypto operation metric collection
 *
 * @param op_name Operation name (must match begin)
 */
void dsssl_crypto_metric_end(const char *op_name);

/**
 * Record crypto metric values
 *
 * @param op_name Operation name
 * @param branches Branch count
 * @param loads Load count
 * @param stores Store count
 * @param cycles Cycle count (0 if not measured)
 */
void dsssl_crypto_metric_record(const char *op_name, uint32_t branches,
                                uint32_t loads, uint32_t stores, uint64_t cycles);

/** @} */

/**
 * @name API Misuse Detection
 * @{
 */

/**
 * Report API misuse
 *
 * @param api API name (e.g., "AEAD_init")
 * @param reason Misuse reason (e.g., "nonce_reuse")
 * @param context_id Context ID
 */
void dsssl_api_misuse_report(const char *api, const char *reason, uint64_t context_id);

/** @} */

/**
 * @name Ticket/PSK/0-RTT Events
 * @{
 */

/**
 * Record ticket event
 *
 * @param subtype Ticket event subtype
 * @param ticket_id Ticket identifier
 */
void dsssl_ticket_event(dsssl_ticket_event_t subtype, uint64_t ticket_id);

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
size_t dsssl_fuzz_get_events(dsssl_telemetry_event_t *events, size_t max_events);

/**
 * Flush telemetry events to file
 *
 * @param filepath Output file path
 * @return 0 on success, negative on error
 */
int dsssl_fuzz_flush_events(const char *filepath);

/**
 * Clear telemetry ring buffer
 */
void dsssl_fuzz_clear_events(void);

/** @} */

/**
 * @name Budget Enforcement
 * @{
 */

/**
 * Check crypto operation budget
 *
 * @param op_name Operation name
 * @param branches Branch count
 * @param loads Load count
 * @param stores Store count
 * @param cycles Cycle count
 * @return 0 if within budget, 1 if violated
 */
int dsssl_crypto_check_budget(const char *op_name, uint32_t branches,
                              uint32_t loads, uint32_t stores, uint64_t cycles);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSSSL_FUZZ_TELEMETRY_H */
