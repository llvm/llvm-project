/**
 * @file dsmil_ot_telemetry.h
 * @brief DSLLVM OT Telemetry Runtime API
 *
 * Provides telemetry functions specifically for Operational Technology (OT)
 * and Industrial Control Systems (ICS) safety monitoring. Focused on:
 * - OT/AI safety boundaries
 * - Layer/device/mission profile awareness
 * - Binary provenance + authority levels
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_OT_TELEMETRY_H
#define DSMIL_OT_TELEMETRY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_OT_TELEMETRY_API OT Telemetry API
 * @{
 */

/**
 * OT telemetry event types
 */
typedef enum {
    DSMIL_TELEMETRY_OT_PATH_ENTRY = 1,  /**< OT-critical function entry */
    DSMIL_TELEMETRY_OT_PATH_EXIT  = 2,  /**< OT-critical function exit */
    DSMIL_TELEMETRY_SES_INTENT    = 3,  /**< SES intent sent */
    DSMIL_TELEMETRY_SES_ACCEPT    = 4,  /**< SES intent accepted */
    DSMIL_TELEMETRY_SES_REJECT    = 5,  /**< SES intent rejected */
    DSMIL_TELEMETRY_INVARIANT_HIT = 6,  /**< Safety invariant checked (passed) */
    DSMIL_TELEMETRY_INVARIANT_FAIL = 7, /**< Safety invariant violation */

    // Telecom / SS7 / SIGTRAN event types
    DSMIL_TELEMETRY_SS7_MSG_RX = 20,      /**< SS7 message received */
    DSMIL_TELEMETRY_SS7_MSG_TX = 21,      /**< SS7 message transmitted */
    DSMIL_TELEMETRY_SIGTRAN_MSG_RX = 22,  /**< SIGTRAN message received */
    DSMIL_TELEMETRY_SIGTRAN_MSG_TX = 23,  /**< SIGTRAN message transmitted */
    DSMIL_TELEMETRY_SIG_ANOMALY = 24      /**< Signaling anomaly detected */
} dsmil_telemetry_event_type_t;

/**
 * OT telemetry event structure
 */
typedef struct {
    dsmil_telemetry_event_type_t event_type;  /**< Event type */
    const char *module_id;                    /**< Object/binary ID (hash or name) */
    const char *func_id;                      /**< Function name */
    const char *file;                         /**< Source file */
    uint32_t    line;                         /**< Source line number */
    uint8_t     layer;                        /**< DSMIL layer (0-8) */
    uint8_t     device;                       /**< DSMIL device (0-103) */
    const char *stage;                        /**< MLOps stage (from DSMIL_STAGE) */
    const char *mission_profile;              /**< Mission profile name */
    uint8_t     authority_tier;               /**< Authority tier (0-3, from DSMIL_OT_TIER) */
    uint64_t    build_id;                     /**< Build ID from DSLLVM provenance */
    uint64_t    provenance_id;                /**< CNSA2 signature ID/hash */

    // Optional numeric payload for safety signals
    const char *signal_name;                  /**< Safety signal name (from DSMIL_SAFETY_SIGNAL) */
    double      signal_value;                 /**< Current signal value */
    double      signal_min;                   /**< Minimum allowed value */
    double      signal_max;                   /**< Maximum allowed value */

    // Optional telecom / SS7 / SIGTRAN fields
    const char *telecom_stack;                /**< Telecom stack: "ss7", "sigtran", "sip", "diameter" */
    const char *ss7_role;                     /**< SS7 role: "STP", "MSC", "HLR", "VLR", "SMSC", "GWMSC", "IN", "GMSC" */
    const char *sigtran_role;                 /**< SIGTRAN role: "SG", "AS", "ASP", "IPSP" */
    const char *telecom_env;                  /**< Environment: "prod", "lab", "honeypot", "fuzz", "sim" */
    const char *telecom_if;                   /**< Interface: "e1", "t1", "sctp", "m2pa", "m2ua", "m3ua", "sua" */
    const char *telecom_ep;                   /**< Logical endpoint (e.g., "upstream_stp", "core_msc") */

    // High-level signaling context (if available)
    uint32_t    ss7_opc;                     /**< SS7 Originating Point Code */
    uint32_t    ss7_dpc;                     /**< SS7 Destination Point Code */
    uint8_t     ss7_sio;                     /**< SS7 Service Information Octet */
    uint32_t    sigtran_rctx;                 /**< SIGTRAN Routing Context (M3UA/SUA), 0 if not set */
    uint8_t     ss7_msg_class;                /**< MTP3/TCAP/CAP message class (if mapped) */
    uint8_t     ss7_msg_type;                 /**< Message type (approximate mapping) */
} dsmil_telemetry_event_t;

/**
 * Log OT telemetry event
 *
 * @param ev Event structure with all metadata
 *
 * This function is called by instrumented code to log OT telemetry events.
 * The implementation is async-safe and uses a ring buffer or simple logging
 * to minimize runtime overhead.
 *
 * Example:
 * @code
 * dsmil_telemetry_event_t ev = {
 *     .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
 *     .module_id = "pump_controller",
 *     .func_id = "pump_control_update",
 *     .file = "pump.c",
 *     .line = 42,
 *     .layer = 3,
 *     .device = 12,
 *     .stage = "control",
 *     .mission_profile = "ics_ops",
 *     .authority_tier = 1,
 *     .build_id = 0x12345678,
 *     .provenance_id = 0xabcdef00,
 *     .signal_name = NULL,
 *     .signal_value = 0.0,
 *     .signal_min = 0.0,
 *     .signal_max = 0.0
 * };
 * dsmil_telemetry_event(&ev);
 * @endcode
 *
 * @note Thread-safe
 * @note Zero overhead if DSMIL_OT_TELEMETRY=0 environment variable is set
 * @note Default implementation writes to stderr in JSON line format
 */
void dsmil_telemetry_event(const dsmil_telemetry_event_t *ev);

/**
 * Log safety signal update
 *
 * @param ev Event structure with signal_name, signal_value, signal_min, signal_max filled
 *
 * Specialized function for logging safety signal updates (pressure, flow,
 * current, speed, etc.). Automatically called by instrumentation when
 * DSMIL_SAFETY_SIGNAL variables are updated.
 *
 * Example:
 * @code
 * dsmil_telemetry_event_t ev = {
 *     .event_type = DSMIL_TELEMETRY_INVARIANT_HIT,
 *     .signal_name = "line7_pressure_setpoint",
 *     .signal_value = 125.5,
 *     .signal_min = 50.0,
 *     .signal_max = 200.0,
 *     .layer = 3,
 *     .device = 12,
 *     .file = "pump.c",
 *     .line = 67
 * };
 * dsmil_telemetry_safety_signal_update(&ev);
 * @endcode
 *
 * @note Thread-safe
 * @note Zero overhead if DSMIL_OT_TELEMETRY=0
 */
void dsmil_telemetry_safety_signal_update(const dsmil_telemetry_event_t *ev);

/**
 * Initialize OT telemetry subsystem
 *
 * Called automatically at program startup if telemetry is enabled.
 * Can be called manually to configure telemetry behavior.
 *
 * @return 0 on success, negative on error
 */
int dsmil_ot_telemetry_init(void);

/**
 * Shutdown OT telemetry subsystem
 *
 * Flushes any pending events and releases resources.
 */
void dsmil_ot_telemetry_shutdown(void);

/**
 * Check if OT telemetry is enabled
 *
 * @return 1 if enabled, 0 if disabled
 *
 * Checks DSMIL_OT_TELEMETRY environment variable (default: ON in production,
 * OFF in tests if desired).
 */
int dsmil_ot_telemetry_is_enabled(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_OT_TELEMETRY_H */
