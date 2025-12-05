/**
 * @file telecom_ss7_example.c
 * @brief Example demonstrating SS7/SIGTRAN telemetry and flagging
 *
 * This example shows how to use DSMIL telecom attributes to mark SS7/SIGTRAN
 * code for compile-time manifest generation and runtime telemetry.
 *
 * Compile with:
 *   dsmil-clang -fdsmil-telecom-flags -fdsmil-mission-profile=ss7_lab \
 *                -c telecom_ss7_example.c -o telecom_ss7_example.o
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_telecom_log.h"
#include "dsmil/include/dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

/**
 * SS7 MTP3 processing function (STP role)
 * 
 * This function processes SS7 MTP3 messages in a Signal Transfer Point (STP).
 * It's marked with:
 * - SS7 stack identification
 * - STP role
 * - Laboratory environment
 * - Defense lab security level
 */
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
DSMIL_STAGE("signaling")
void ss7_mtp3_process(const uint8_t *msg, size_t len) {
    if (len < 5) {
        return;  // Invalid message
    }

    // Extract SS7 header fields
    uint32_t opc = (msg[0] << 16) | (msg[1] << 8) | msg[2];
    uint32_t dpc = (msg[3] << 16) | (msg[4] << 8) | msg[5];
    uint8_t sio = msg[6];
    uint8_t msg_class = msg[7];
    uint8_t msg_type = msg[8];

    // Log SS7 message received
    DSMIL_LOG_SS7_RX(opc, dpc, sio, msg_class, msg_type);

    // Process MTP3 message
    printf("SS7 MTP3: OPC=%u DPC=%u SIO=0x%02x\n", opc, dpc, sio);

    // Routing logic would go here...
}

/**
 * SIGTRAN M3UA processing function (Signaling Gateway role)
 * 
 * Processes SIGTRAN M3UA messages in a Signaling Gateway.
 */
DSMIL_TELECOM_STACK("sigtran")
DSMIL_SIGTRAN_ROLE("SG")
DSMIL_TELECOM_INTERFACE("m3ua")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(32)
DSMIL_STAGE("signaling")
void sigtran_m3ua_rx(const uint8_t *msg, size_t len, uint32_t rctx) {
    // Log SIGTRAN message received
    DSMIL_LOG_SIGTRAN_RX(rctx);

    printf("SIGTRAN M3UA: Routing Context=%u\n", rctx);

    // Convert SIGTRAN to SS7 and forward
    // (simplified - real implementation would do proper conversion)
}

/**
 * Honeypot SS7 handler
 * 
 * This function is marked as honeypot code and must NOT run in production.
 * The compiler will enforce this via mission profile checks.
 */
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("honeypot")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_TELECOM_ENDPOINT("honeypot_stp")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
DSMIL_STAGE("signaling")
void honeypot_ss7_handler(const uint8_t *msg, size_t len) {
    // Honeypot SS7 handler - logs all messages for analysis
    printf("Honeypot SS7: Received %zu bytes\n", len);

    // Log anomaly if suspicious pattern detected
    if (len > 1000) {  // Suspiciously large message
        DSMIL_LOG_SIG_ANOMALY("ss7", "Oversized SS7 message");
    }
}

/**
 * Production MSC handler
 * 
 * Production code for Mobile Switching Center.
 */
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("MSC")
DSMIL_TELECOM_ENV("prod")
DSMIL_SIG_SECURITY("high_assurance")
DSMIL_TELECOM_ENDPOINT("core_msc")
DSMIL_LAYER(3)
DSMIL_DEVICE(33)
DSMIL_STAGE("signaling")
void prod_msc_handler(const uint8_t *msg, size_t len) {
    // Production MSC handler
    uint32_t opc = 0, dpc = 0;
    uint8_t sio = 0, msg_class = 0, msg_type = 0;

    if (len >= 9) {
        opc = (msg[0] << 16) | (msg[1] << 8) | msg[2];
        dpc = (msg[3] << 16) | (msg[4] << 8) | msg[5];
        sio = msg[6];
        msg_class = msg[7];
        msg_type = msg[8];
    }

    // Log with full context
    DSMIL_LOG_SS7_FULL(opc, dpc, sio, msg_class, msg_type, "MSC", "prod");

    printf("Production MSC: Processing message\n");
}

/**
 * Fuzzing interface handler
 * 
 * Code for fuzzing SS7 implementations.
 */
DSMIL_TELECOM_STACK("ss7")
DSMIL_TELECOM_ENV("fuzz")
DSMIL_SIG_SECURITY("low")
DSMIL_TELECOM_INTERFACE("e1")
DSMIL_LAYER(3)
DSMIL_DEVICE(34)
DSMIL_STAGE("signaling")
void fuzz_ss7_handler(const uint8_t *msg, size_t len) {
    // Fuzzing handler - must not run in production
    printf("Fuzz SS7: Testing with %zu bytes\n", len);
}

/**
 * Main function
 */
int main(int argc, char **argv) {
    // Initialize telemetry
    dsmil_ot_telemetry_init();

    printf("SS7/SIGTRAN Telecom Example\n");
    printf("===========================\n\n");

    // Example SS7 message (simplified)
    uint8_t ss7_msg[] = {
        0x00, 0x01, 0x02,  // OPC
        0x00, 0x03, 0x04,  // DPC
        0x08,              // SIO
        0x01,              // Message class
        0x02               // Message type
    };

    // Call SS7 handler
    ss7_mtp3_process(ss7_msg, sizeof(ss7_msg));

    // Call SIGTRAN handler
    sigtran_m3ua_rx(ss7_msg, sizeof(ss7_msg), 100);

    // Call honeypot handler (only in honeypot environment)
    honeypot_ss7_handler(ss7_msg, sizeof(ss7_msg));

    printf("\nTelecom manifest should be generated: telecom_ss7_example.dsmil.telecom.json\n");
    printf("Telemetry events logged to stderr (if DSMIL_OT_TELEMETRY=1)\n");

    dsmil_ot_telemetry_shutdown();
    return 0;
}
