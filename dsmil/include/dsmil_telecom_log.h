/**
 * @file dsmil_telecom_log.h
 * @brief DSLLVM Telecom Telemetry Helper Macros
 *
 * Provides convenient helper macros for telecom-aware telemetry logging.
 * Simplifies integration with miltop_ss7, OSMOCOM-based code, and other
 * telecom modules.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_TELECOM_LOG_H
#define DSMIL_TELECOM_LOG_H

#include "dsmil_ot_telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_TELECOM_LOG Telecom Telemetry Helpers
 * @{
 */

/**
 * Log SS7 message received event
 *
 * @param opc Originating Point Code
 * @param dpc Destination Point Code
 * @param sio Service Information Octet
 * @param msg_class Message class (MTP3/TCAP/CAP)
 * @param msg_type Message type
 *
 * Example:
 * @code
 * void ss7_mtp3_rx(uint32_t opc, uint32_t dpc, uint8_t sio, uint8_t msg_class, uint8_t msg_type) {
 *     DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type);
 *     // Process message...
 * }
 * @endcode
 */
#define DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SS7_MSG_RX; \
        ev.ss7_opc = (opc); \
        ev.ss7_dpc = (dpc); \
        ev.ss7_sio = (sio); \
        ev.ss7_msg_class = (msg_class); \
        ev.ss7_msg_type = (msg_type); \
        ev.telecom_stack = "ss7"; \
        dsmil_telemetry_event(&ev); \
    } while (0)

/**
 * Log SS7 message transmitted event
 *
 * @param opc Originating Point Code
 * @param dpc Destination Point Code
 * @param sio Service Information Octet
 * @param msg_class Message class
 * @param msg_type Message type
 */
#define DSMIL_LOG_SS7_TX(opc, dpc, sio, msg_class, msg_type) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SS7_MSG_TX; \
        ev.ss7_opc = (opc); \
        ev.ss7_dpc = (dpc); \
        ev.ss7_sio = (sio); \
        ev.ss7_msg_class = (msg_class); \
        ev.ss7_msg_type = (msg_type); \
        ev.telecom_stack = "ss7"; \
        dsmil_telemetry_event(&ev); \
    } while (0)

/**
 * Log SIGTRAN message received event
 *
 * @param rctx Routing Context (M3UA/SUA), 0 if not applicable
 *
 * Example:
 * @code
 * void sigtran_m3ua_rx(uint32_t rctx) {
 *     DSMIL_LOG_SIGTRAN_RX(rctx);
 *     // Process SIGTRAN message...
 * }
 * @endcode
 */
#define DSMIL_LOG_SIGTRAN_RX(rctx) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SIGTRAN_MSG_RX; \
        ev.sigtran_rctx = (rctx); \
        ev.telecom_stack = "sigtran"; \
        dsmil_telemetry_event(&ev); \
    } while (0)

/**
 * Log SIGTRAN message transmitted event
 *
 * @param rctx Routing Context (M3UA/SUA), 0 if not applicable
 */
#define DSMIL_LOG_SIGTRAN_TX(rctx) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SIGTRAN_MSG_TX; \
        ev.sigtran_rctx = (rctx); \
        ev.telecom_stack = "sigtran"; \
        dsmil_telemetry_event(&ev); \
    } while (0)

/**
 * Log signaling anomaly event
 *
 * @param stack Telecom stack ("ss7", "sigtran", etc.)
 * @param description Anomaly description (optional, can be NULL)
 *
 * Example:
 * @code
 * if (unusual_pattern_detected) {
 *     DSMIL_LOG_SIG_ANOMALY("ss7", "Unexpected message sequence");
 * }
 * @endcode
 */
#define DSMIL_LOG_SIG_ANOMALY(stack, description) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SIG_ANOMALY; \
        ev.telecom_stack = (stack); \
        /* Note: description could be added to a message field if extended */ \
        dsmil_telemetry_event(&ev); \
    } while (0)

/**
 * Log SS7 message with full context
 *
 * @param opc Originating Point Code
 * @param dpc Destination Point Code
 * @param sio Service Information Octet
 * @param msg_class Message class
 * @param msg_type Message type
 * @param role SS7 role ("STP", "MSC", etc.)
 * @param env Environment ("prod", "lab", "honeypot", etc.)
 *
 * Full-featured logging with role and environment context.
 */
#define DSMIL_LOG_SS7_FULL(opc, dpc, sio, msg_class, msg_type, role, env) \
    do { \
        dsmil_telemetry_event_t ev = {0}; \
        ev.event_type = DSMIL_TELEMETRY_SS7_MSG_RX; \
        ev.ss7_opc = (opc); \
        ev.ss7_dpc = (dpc); \
        ev.ss7_sio = (sio); \
        ev.ss7_msg_class = (msg_class); \
        ev.ss7_msg_type = (msg_type); \
        ev.telecom_stack = "ss7"; \
        ev.ss7_role = (role); \
        ev.telecom_env = (env); \
        dsmil_telemetry_event(&ev); \
    } while (0)

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_TELECOM_LOG_H */
