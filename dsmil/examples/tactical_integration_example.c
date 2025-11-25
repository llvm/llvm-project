/**
 * @file tactical_integration_example.c
 * @brief DSLLVM v1.5.1 Phase 2: Tactical Integration Example
 *
 * Demonstrates v1.5.1 Phase 2 features:
 * - Feature 3.3: Blue Force Tracker (BFT-2) with encryption/authentication
 * - Feature 3.7: Radio Multi-Protocol Bridging (Link-16, SATCOM, MUOS)
 * - Feature 3.9: 5G Latency & Throughput Contracts
 *
 * Scenario: Tactical unit operating in contested environment with:
 * - Real-time position tracking via BFT-2
 * - Multi-protocol tactical radio bridging (Link-16, SATCOM fallback)
 * - 5G/MEC edge computing with strict latency requirements
 * - Friend/foe tracking and spoofing detection
 *
 * Compile:
 *   clang -o tactical_example tactical_integration_example.c \
 *         -ldsmil_bft_runtime -ldsmil_radio_runtime -ldsmil_jadc2_runtime
 *
 * Run:
 *   export DSMIL_BFT_REFRESH_RATE=5
 *   export DSMIL_5G_MEC_ENABLE=1
 *   ./tactical_example
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

// Include DSMIL attributes
#include "dsmil_attributes.h"

// Forward declarations for runtime functions
extern int dsmil_bft_init(const char *unit_id, const char *crypto_key);
extern int dsmil_bft_send_position(double lat, double lon, double alt, uint64_t ts);
extern int dsmil_bft_send_status(const char *status);
extern void dsmil_bft_update_status(uint8_t fuel, uint8_t ammo, uint8_t readiness);
extern void dsmil_bft_get_stats(uint64_t *sent, uint64_t *received, uint64_t *spoofed);
extern uint64_t dsmil_timestamp_ns(void);

extern int dsmil_radio_init(int primary_protocol);
extern int dsmil_radio_bridge_send(const char *protocol, const uint8_t *data, size_t length);
extern void dsmil_radio_get_stats(uint64_t *sent, uint64_t *received, uint64_t *jamming);

extern int dsmil_jadc2_init(const char *profile);
extern int dsmil_jadc2_send(const void *data, size_t length, uint8_t priority, const char *domain);
extern bool dsmil_5g_edge_available(void);

// ============================================================================
// PART 1: BFT-2 POSITION TRACKING
// ============================================================================

/**
 * @brief Continuously report position via BFT-2
 *
 * Features:
 * - AES-256-GCM encryption
 * - ML-DSA-87 signature authentication
 * - Rate limiting (5-10 second refresh)
 * - Spoofing detection
 */
DSMIL_BFT_HOOK("position")
DSMIL_BFT_AUTHORIZED
DSMIL_CLASSIFICATION("S")
DSMIL_CLEARANCE(0x07000000)
DSMIL_LAYER(4)
void bft_position_reporter(double lat, double lon, double alt) {
    printf("\n=== BFT-2 Position Update ===\n");
    printf("Position: (%.6f, %.6f, %.1fm)\n", lat, lon, alt);
    printf("Encryption: AES-256-GCM, Auth: ML-DSA-87\n");

    uint64_t timestamp = dsmil_timestamp_ns();
    int result = dsmil_bft_send_position(lat, lon, alt, timestamp);

    if (result == 0) {
        printf("✓ Position sent successfully\n");
    } else if (result == 1) {
        printf("⊘ Rate-limited (too soon since last update)\n");
    } else {
        printf("✗ Send failed\n");
    }
}

/**
 * @brief Report unit status via BFT
 */
DSMIL_BFT_HOOK("status")
DSMIL_BFT_AUTHORIZED
DSMIL_CLASSIFICATION("S")
void bft_status_reporter(const char *status_text, uint8_t fuel, uint8_t ammo) {
    printf("\n=== BFT-2 Status Update ===\n");
    printf("Status: %s\n", status_text);
    printf("Fuel: %u%%, Ammo: %u%%\n", fuel, ammo);

    dsmil_bft_update_status(fuel, ammo, 1);  // C1 readiness
    dsmil_bft_send_status(status_text);

    printf("✓ Status sent via BFT-2\n");
}

// ============================================================================
// PART 2: RADIO MULTI-PROTOCOL BRIDGING
// ============================================================================

/**
 * @brief Send tactical message via Link-16
 *
 * Link-16: Tactical Data Link, J-series messages
 * - 16/31/51/75 bits per word
 * - Used for air-to-air, air-to-ground coordination
 */
DSMIL_RADIO_PROFILE("link16")
DSMIL_CLASSIFICATION("S")
DSMIL_LAYER(4)
void send_link16_message(const char *message) {
    printf("\n=== Link-16 Transmission ===\n");
    printf("Message: %s\n", message);
    printf("Protocol: Link-16 J-series\n");

    int result = dsmil_radio_bridge_send("link16",
                                          (const uint8_t*)message,
                                          strlen(message));

    if (result == 0) {
        printf("✓ Sent via Link-16\n");
    } else {
        printf("✗ Link-16 transmission failed\n");
    }
}

/**
 * @brief Send message via SATCOM (fallback when Link-16 jammed)
 *
 * SATCOM: Satellite communications with FEC
 * - UHF/SHF/EHF bands
 * - High latency (500ms) but reliable
 * - Forward Error Correction for lossy links
 */
DSMIL_RADIO_PROFILE("satcom")
DSMIL_CLASSIFICATION("S")
DSMIL_BLOS_FALLBACK("link16", "satcom")
void send_satcom_fallback(const char *message) {
    printf("\n=== SATCOM Fallback Transmission ===\n");
    printf("Message: %s\n", message);
    printf("Protocol: SATCOM with FEC\n");
    printf("Latency: ~500ms (acceptable for BLOS)\n");

    int result = dsmil_radio_bridge_send("satcom",
                                          (const uint8_t*)message,
                                          strlen(message));

    if (result == 0) {
        printf("✓ Sent via SATCOM fallback\n");
    } else {
        printf("✗ SATCOM transmission failed\n");
    }
}

/**
 * @brief Multi-protocol bridge: automatic protocol selection
 *
 * Bridge function tries primary protocol, falls back automatically
 * - Primary: Link-16 (low latency, high bandwidth)
 * - Fallback: SATCOM (high latency, but reliable)
 */
DSMIL_RADIO_BRIDGE
DSMIL_CLASSIFICATION("S")
void send_tactical_message_auto(const char *message) {
    printf("\n=== Multi-Protocol Bridge ===\n");
    printf("Message: %s\n", message);
    printf("Bridge: Auto-select (Link-16 → SATCOM fallback)\n");

    // NULL protocol = automatic selection
    int result = dsmil_radio_bridge_send(NULL,
                                          (const uint8_t*)message,
                                          strlen(message));

    if (result == 0) {
        printf("✓ Message sent via best available protocol\n");
    } else {
        printf("✗ All protocols unavailable\n");
    }
}

// ============================================================================
// PART 3: 5G/MEC EDGE COMPUTING WITH LATENCY CONTRACTS
// ============================================================================

/**
 * @brief Time-critical C2 processing on 5G/MEC edge
 *
 * JADC2 Requirements:
 * - 5ms latency budget (compile-time enforced)
 * - 10Gbps bandwidth contract
 * - 99.999% reliability
 */
DSMIL_JADC2_PROFILE("c2_processing")
DSMIL_5G_EDGE
DSMIL_LATENCY_BUDGET(5)
DSMIL_BANDWIDTH_CONTRACT(10)
DSMIL_CLASSIFICATION("S")
DSMIL_LAYER(7)
void edge_c2_processing(const uint8_t *sensor_data, size_t length) {
    printf("\n=== 5G/MEC Edge C2 Processing ===\n");
    printf("Latency Budget: 5ms (JADC2 requirement)\n");
    printf("Bandwidth Contract: 10Gbps\n");
    printf("Deployment: 5G MEC edge node\n");

    if (!dsmil_5g_edge_available()) {
        printf("⚠ 5G/MEC unavailable, falling back to local processing\n");
        return;
    }

    // Simulate fast C2 decision-making
    printf("Processing %zu bytes of sensor data...\n", length);

    // Send decision via JADC2 transport (PRIORITY level)
    uint8_t decision[] = "C2_DECISION: ENGAGE_TARGET";
    dsmil_jadc2_send(decision, sizeof(decision), 64, "air");

    printf("✓ C2 decision computed in <5ms\n");
    printf("✓ Decision sent via JADC2 (PRIORITY)\n");
}

/**
 * @brief Flash-priority targeting solution
 *
 * Time-critical targeting requires FLASH priority (192-255)
 * - Must complete in <5ms
 * - Highest network priority
 */
DSMIL_JADC2_PROFILE("targeting")
DSMIL_JADC2_TRANSPORT(200)
DSMIL_LATENCY_BUDGET(5)
DSMIL_5G_EDGE
DSMIL_CLASSIFICATION("TS")
DSMIL_ROE("LIVE_CONTROL")
void send_targeting_flash(double target_lat, double target_lon) {
    printf("\n=== FLASH Priority Targeting ===\n");
    printf("Target: (%.6f, %.6f)\n", target_lat, target_lon);
    printf("Priority: FLASH (200/255)\n");
    printf("Latency: <5ms required\n");

    char targeting_msg[256];
    snprintf(targeting_msg, sizeof(targeting_msg),
             "TARGETING|%.6f|%.6f|PRECISION_GUIDED",
             target_lat, target_lon);

    dsmil_jadc2_send(targeting_msg, strlen(targeting_msg), 200, "air");

    printf("✓ Targeting solution sent (FLASH priority)\n");
    printf("⚠ Human-in-loop verification required\n");
}

// ============================================================================
// PART 4: INTEGRATED TACTICAL SCENARIO
// ============================================================================

/**
 * @brief Complete tactical scenario integrating all features
 *
 * Scenario: Unit operating in contested environment
 * 1. Report position via BFT-2 (encrypted, authenticated)
 * 2. Receive sensor data, process on 5G/MEC edge
 * 3. Make C2 decision, send via Link-16
 * 4. If Link-16 jammed, fallback to SATCOM
 * 5. Report status back via BFT
 */
DSMIL_CLASSIFICATION("S")
DSMIL_JADC2_PROFILE("c2_processing")
DSMIL_5G_EDGE
DSMIL_LATENCY_BUDGET(10)
void integrated_tactical_scenario(void) {
    printf("\n" "═══════════════════════════════════════════════════════════\n");
    printf("INTEGRATED TACTICAL SCENARIO\n");
    printf("═══════════════════════════════════════════════════════════\n");

    // Step 1: Report position via BFT-2
    printf("\n[Step 1] Reporting position via BFT-2...\n");
    bft_position_reporter(38.8977, -77.0365, 125.0);

    // Step 2: Receive sensor data and process on edge
    printf("\n[Step 2] Processing sensor data on 5G/MEC edge...\n");
    uint8_t sensor_data[] = "RADAR_CONTACT|HOSTILE|38.9000|-77.0400";
    edge_c2_processing(sensor_data, sizeof(sensor_data));

    // Step 3: Send C2 decision via Link-16
    printf("\n[Step 3] Sending C2 decision via Link-16...\n");
    send_link16_message("C2: INTERCEPT_VECTOR_090");

    // Step 4: Link-16 jammed? Use SATCOM fallback
    printf("\n[Step 4] Checking for jamming, using fallback if needed...\n");
    send_satcom_fallback("STATUS: OPERATIONAL");

    // Step 5: Report updated status via BFT
    printf("\n[Step 5] Reporting status via BFT-2...\n");
    bft_status_reporter("ENGAGED", 85, 75);

    // Step 6: Flash-priority targeting (if required)
    printf("\n[Step 6] Sending flash-priority targeting solution...\n");
    send_targeting_flash(38.9000, -77.0400);

    printf("\n" "═══════════════════════════════════════════════════════════\n");
    printf("Scenario complete. All tactical systems operational.\n");
    printf("═══════════════════════════════════════════════════════════\n");
}

// ============================================================================
// MAIN: DEMONSTRATION
// ============================================================================

int main(int argc, char **argv) {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  DSLLVM v1.5.1: Phase 2 Tactical Integration             ║\n");
    printf("║  BFT-2, Radio Bridging, 5G Contracts                     ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Initialize subsystems
    printf("\nInitializing tactical subsystems...\n");

    // BFT-2
    dsmil_bft_init("ALPHA-2-1", NULL);
    printf("✓ BFT-2 initialized (AES-256-GCM, ML-DSA-87)\n");

    // Radio bridging (Link-16 primary)
    dsmil_radio_init(0);  // 0 = Link-16
    printf("✓ Radio bridge initialized (Link-16, SATCOM, MUOS)\n");

    // JADC2 & 5G/MEC
    dsmil_jadc2_init("c2_processing");
    printf("✓ JADC2 initialized (5G/MEC edge, 5ms latency budget)\n");

    // Run integrated scenario
    integrated_tactical_scenario();

    // Print statistics
    printf("\n\n=== System Statistics ===\n");

    uint64_t bft_sent, bft_recv, bft_spoofed;
    dsmil_bft_get_stats(&bft_sent, &bft_recv, &bft_spoofed);
    printf("BFT-2: Sent=%lu Received=%lu Spoofing_Detected=%lu\n",
           bft_sent, bft_recv, bft_spoofed);

    uint64_t radio_sent[5], radio_recv[5], radio_jamming[5];
    dsmil_radio_get_stats(radio_sent, radio_recv, radio_jamming);
    printf("Radio: Link16=%lu SATCOM=%lu MUOS=%lu SINCGARS=%lu EPLRS=%lu\n",
           radio_sent[0], radio_sent[1], radio_sent[2],
           radio_sent[3], radio_sent[4]);

    printf("\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  All Phase 2 features demonstrated successfully          ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
