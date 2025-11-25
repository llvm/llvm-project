/**
 * @file jadc2_cross_domain_example.c
 * @brief DSLLVM v1.5 Comprehensive Example: JADC2 + Cross-Domain Security
 *
 * This example demonstrates:
 * 1. Classification-aware cross-domain security
 * 2. JADC2 sensor→C2→shooter pipeline
 * 3. 5G/MEC edge deployment
 * 4. Blue Force Tracker (BFT) integration
 * 5. Resilient communications (BLOS fallback)
 *
 * Scenario: Multi-domain C2 system processing classified sensor data,
 * making targeting decisions, and coordinating with coalition partners.
 *
 * Compile:
 *   clang -o jadc2_example jadc2_cross_domain_example.c \
 *         -ldsmil_cross_domain_runtime -ldsmil_jadc2_runtime
 *
 * Run:
 *   # SECRET network (SIPRNET)
 *   export DSMIL_NETWORK_CLASSIFICATION=S
 *   export DSMIL_5G_MEC_ENABLE=1
 *   ./jadc2_example
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// Include DSMIL attributes
#include "dsmil_attributes.h"

// Forward declarations for runtime functions
extern int dsmil_cross_domain_init(const char *network_classification);
extern int dsmil_cross_domain_guard(const void *data, size_t length,
                                      const char *from, const char *to,
                                      const char *policy);
extern int dsmil_jadc2_init(const char *profile);
extern int dsmil_jadc2_send(const void *data, size_t length,
                             uint8_t priority, const char *domain);
extern int dsmil_bft_init(const char *unit_id, const char *crypto_key);
extern int dsmil_bft_send_position(double lat, double lon, double alt,
                                     uint64_t timestamp_ns);
extern uint64_t dsmil_timestamp_ns(void);
extern bool dsmil_5g_edge_available(void);
extern int dsmil_resilient_send(const void *data, size_t length);
extern void dsmil_emcon_activate(uint8_t level);

// ============================================================================
// PART 1: CLASSIFIED SENSOR DATA PROCESSING
// ============================================================================

// Sensor data structure (SECRET classification)
typedef struct {
    double latitude;
    double longitude;
    char target_type[64];
    float confidence;
    uint64_t timestamp;
} sensor_reading_t;

/**
 * @brief Process SECRET sensor data (radar, EO/IR, SIGINT)
 *
 * Classification: SECRET (SIPRNET)
 * JADC2 Profile: sensor_fusion
 * Latency Budget: 5ms (JADC2 requirement)
 */
DSMIL_CLASSIFICATION("S")
DSMIL_JADC2_PROFILE("sensor_fusion")
DSMIL_LATENCY_BUDGET(5)
DSMIL_5G_EDGE
DSMIL_LAYER(7)
void process_sensor_data_secret(const sensor_reading_t *readings,
                                  size_t count) {
    printf("\n=== SECRET Sensor Fusion ===\n");
    printf("Processing %zu sensor readings (5G/MEC edge)\n", count);

    for (size_t i = 0; i < count; i++) {
        printf("  Sensor %zu: %s at (%.4f, %.4f) confidence=%.2f\n",
               i,
               readings[i].target_type,
               readings[i].latitude,
               readings[i].longitude,
               readings[i].confidence);
    }

    // Send fused data via JADC2 transport (SECRET→C2)
    dsmil_jadc2_send(readings, count * sizeof(sensor_reading_t),
                     128,  // IMMEDIATE priority
                     "air");

    printf("Sensor data sent to C2 via JADC2 (SECRET)\n");
}

// ============================================================================
// PART 2: CROSS-DOMAIN DOWNGRADE (SECRET → CONFIDENTIAL)
// ============================================================================

// Sanitized target data (CONFIDENTIAL classification)
typedef struct {
    double latitude;
    double longitude;
    char target_category[32];  // Sanitized: no specific type
    uint64_t timestamp;
} sanitized_target_t;

/**
 * @brief Cross-domain gateway: Downgrade SECRET→CONFIDENTIAL
 *
 * Implements sanitization and guard policy for classification downgrade.
 * Required for releasing data to coalition partners (MPE).
 */
DSMIL_CROSS_DOMAIN_GATEWAY("S", "C")
DSMIL_GUARD_APPROVED
DSMIL_LAYER(8)  // Security AI layer validates sanitization
int sanitize_target_data(const sensor_reading_t *secret_data,
                          size_t count,
                          sanitized_target_t *confidential_output) {
    printf("\n=== Cross-Domain Sanitization ===\n");
    printf("Downgrading %zu targets: SECRET → CONFIDENTIAL\n", count);

    // Invoke cross-domain guard
    int result = dsmil_cross_domain_guard(
        secret_data,
        count * sizeof(sensor_reading_t),
        "S",   // From SECRET
        "C",   // To CONFIDENTIAL
        "manual_review"  // Guard policy
    );

    if (result != 0) {
        printf("ERROR: Cross-domain guard rejected downgrade!\n");
        return -1;
    }

    // Sanitization: remove sensitive details
    for (size_t i = 0; i < count; i++) {
        confidential_output[i].latitude = secret_data[i].latitude;
        confidential_output[i].longitude = secret_data[i].longitude;
        confidential_output[i].timestamp = secret_data[i].timestamp;

        // Generalize target type (sanitization)
        if (strstr(secret_data[i].target_type, "radar")) {
            strcpy(confidential_output[i].target_category, "GROUND");
        } else {
            strcpy(confidential_output[i].target_category, "UNKNOWN");
        }
    }

    printf("Sanitization complete. Data safe for CONFIDENTIAL release.\n");
    return 0;
}

// ============================================================================
// PART 3: MISSION PARTNER ENVIRONMENT (COALITION SHARING)
// ============================================================================

/**
 * @brief Send sanitized data to NATO partners
 *
 * Classification: CONFIDENTIAL
 * Releasability: REL NATO
 * Mission Partner Environment: Allied networks
 */
DSMIL_CLASSIFICATION("C")
DSMIL_MPE_PARTNER("NATO")
DSMIL_RELEASABILITY("REL NATO")
DSMIL_JADC2_PROFILE("c2_processing")
void share_with_nato(const sanitized_target_t *targets, size_t count) {
    printf("\n=== Mission Partner Environment ===\n");
    printf("Sharing %zu sanitized targets with NATO (CONFIDENTIAL)\n", count);

    for (size_t i = 0; i < count; i++) {
        printf("  Target %zu: %s at (%.4f, %.4f)\n",
               i,
               targets[i].target_category,
               targets[i].latitude,
               targets[i].longitude);
    }

    // Send via MPE cross-domain gateway
    dsmil_jadc2_send(targets, count * sizeof(sanitized_target_t),
                     64,  // PRIORITY (not flash - coalition data)
                     "land");

    printf("Data shared with NATO partners (MPE)\n");
}

// ============================================================================
// PART 4: C2 PROCESSING AND TARGETING (TOP SECRET)
// ============================================================================

// Targeting solution (TOP SECRET classification)
typedef struct {
    double target_lat;
    double target_lon;
    char weapon_type[64];
    uint8_t authorization_code;
} targeting_solution_t;

/**
 * @brief AI-assisted targeting (TOP SECRET, human-in-loop required)
 *
 * Classification: TOP SECRET
 * JADC2 Profile: targeting
 * Transport Priority: FLASH (time-critical)
 */
DSMIL_CLASSIFICATION("TS")
DSMIL_JADC2_PROFILE("targeting")
DSMIL_AUTOTARGET
DSMIL_JADC2_TRANSPORT(200)  // FLASH priority
DSMIL_ROE("LIVE_CONTROL")
DSMIL_LATENCY_BUDGET(5)
DSMIL_LAYER(7)
void autotarget_engage(const sensor_reading_t *sensor_data,
                        float confidence_threshold) {
    printf("\n=== AI-Assisted Targeting (TOP SECRET) ===\n");

    if (sensor_data->confidence < confidence_threshold) {
        printf("Confidence %.2f below threshold %.2f - no engagement\n",
               sensor_data->confidence, confidence_threshold);
        return;
    }

    printf("High-confidence target detected: %s (conf=%.2f)\n",
           sensor_data->target_type, sensor_data->confidence);

    // Generate targeting solution
    targeting_solution_t solution;
    solution.target_lat = sensor_data->latitude;
    solution.target_lon = sensor_data->longitude;
    strcpy(solution.weapon_type, "precision_guided");
    solution.authorization_code = 0xAA;  // Simplified

    // Human-in-loop verification required
    printf("HUMAN VERIFICATION REQUIRED for lethal engagement\n");
    printf("Target: (%.4f, %.4f), Weapon: %s\n",
           solution.target_lat, solution.target_lon, solution.weapon_type);

    // Send to shooter via JADC2 (FLASH priority)
    dsmil_jadc2_send(&solution, sizeof(solution), 200, "air");

    printf("Targeting solution sent to shooter (TOP SECRET, FLASH)\n");
}

// ============================================================================
// PART 5: BLUE FORCE TRACKER (BFT) INTEGRATION
// ============================================================================

/**
 * @brief Report friendly position via BFT
 *
 * Classification: SECRET (position data)
 * BFT-2 protocol: AES-256 encrypted
 */
DSMIL_CLASSIFICATION("S")
DSMIL_BFT_HOOK("position")
DSMIL_BFT_AUTHORIZED
DSMIL_CLEARANCE(0x07000000)
void report_friendly_position(double lat, double lon, double alt) {
    printf("\n=== Blue Force Tracker ===\n");
    printf("Reporting position: (%.6f, %.6f, %.1fm)\n", lat, lon, alt);

    uint64_t timestamp = dsmil_timestamp_ns();
    dsmil_bft_send_position(lat, lon, alt, timestamp);

    printf("Position sent via BFT-2 (AES-256 encrypted)\n");
}

// ============================================================================
// PART 6: RESILIENT COMMUNICATIONS (EMCON & BLOS)
// ============================================================================

/**
 * @brief Covert transmission in contested environment
 *
 * Classification: SECRET
 * EMCON Level: 3 (low signature)
 * BLOS Fallback: 5G → SATCOM
 */
DSMIL_CLASSIFICATION("S")
DSMIL_EMCON_MODE(3)
DSMIL_LOW_SIGNATURE("aggressive")
DSMIL_BLOS_FALLBACK("5g", "satcom")
void covert_transmission(const uint8_t *data, size_t length) {
    printf("\n=== Covert Transmission (EMCON) ===\n");
    printf("EMCON Level 3: Low RF signature, batched transmission\n");

    // Activate EMCON mode
    dsmil_emcon_activate(3);

    // Check if 5G available, fallback to SATCOM if jammed
    if (dsmil_5g_edge_available()) {
        printf("Using primary link: 5G/MEC\n");
    } else {
        printf("Primary jammed, falling back to SATCOM (high latency)\n");
    }

    dsmil_resilient_send(data, length);

    printf("Covert transmission complete\n");
}

// ============================================================================
// PART 7: U.S.-ONLY INTELLIGENCE (NO COALITION RELEASE)
// ============================================================================

/**
 * @brief Process U.S.-only intelligence (not releasable to partners)
 *
 * Classification: TOP SECRET/SCI
 * Releasability: NOFORN (no foreign nationals)
 */
DSMIL_US_ONLY
DSMIL_CLASSIFICATION("TS/SCI")
DSMIL_RELEASABILITY("NOFORN")
DSMIL_LAYER(7)
void process_us_only_intelligence(const char *classified_source) {
    printf("\n=== U.S.-Only Intelligence ===\n");
    printf("Processing TOP SECRET/SCI NOFORN data\n");
    printf("Source: %s\n", classified_source);
    printf("NOT releasable to coalition partners\n");

    // This function cannot be called from MPE partner functions
    // Compile-time error if MPE code tries to call this
}

// ============================================================================
// MAIN: DEMONSTRATION
// ============================================================================

int main(int argc, char **argv) {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  DSLLVM v1.5: JADC2 + Cross-Domain Security Example      ║\n");
    printf("║  War-Fighting Compiler for C3/JADC2 Systems              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Initialize cross-domain guard (SECRET network / SIPRNET)
    const char *network_class = getenv("DSMIL_NETWORK_CLASSIFICATION");
    if (!network_class) {
        network_class = "S";  // Default: SECRET (SIPRNET)
    }
    printf("\nInitializing on %s network...\n", network_class);
    dsmil_cross_domain_init(network_class);

    // Initialize JADC2 transport
    dsmil_jadc2_init("sensor_fusion");

    // Initialize BFT
    dsmil_bft_init("ALPHA-1", NULL);

    // ========================================================================
    // SCENARIO 1: SECRET Sensor Fusion → C2
    // ========================================================================
    printf("\n" "═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 1: Multi-sensor fusion (SECRET)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    sensor_reading_t sensors[3] = {
        {38.8977, -77.0365, "radar_contact", 0.92, 1234567890},
        {38.8980, -77.0370, "eo_ir_signature", 0.87, 1234567891},
        {38.8975, -77.0368, "sigint_intercept", 0.95, 1234567892}
    };

    process_sensor_data_secret(sensors, 3);

    // ========================================================================
    // SCENARIO 2: Cross-Domain Downgrade → Coalition Sharing
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 2: Cross-domain sanitization & MPE sharing\n");
    printf("═══════════════════════════════════════════════════════════\n");

    sanitized_target_t nato_targets[3];
    if (sanitize_target_data(sensors, 3, nato_targets) == 0) {
        share_with_nato(nato_targets, 3);
    }

    // ========================================================================
    // SCENARIO 3: AI-Assisted Targeting (TOP SECRET)
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 3: AI-assisted targeting (TOP SECRET)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    autotarget_engage(&sensors[2], 0.90);

    // ========================================================================
    // SCENARIO 4: Blue Force Tracker
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 4: Blue Force Tracker position reporting\n");
    printf("═══════════════════════════════════════════════════════════\n");

    report_friendly_position(38.8977, -77.0365, 125.0);

    // ========================================================================
    // SCENARIO 5: Covert Operations (EMCON)
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 5: Covert transmission (EMCON + BLOS fallback)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    uint8_t covert_msg[] = "STEALTH_OPS_ACTIVE";
    covert_transmission(covert_msg, sizeof(covert_msg));

    // ========================================================================
    // SCENARIO 6: U.S.-Only Intelligence
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("SCENARIO 6: U.S.-only intelligence (NOFORN)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    process_us_only_intelligence("CLASSIFIED_SOURCE_ALPHA");

    // ========================================================================
    printf("\n\n╔════════════════════════════════════════════════════════════╗\n");
    printf("║  All scenarios complete. DSLLVM v1.5 demonstration done.  ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
