/**
 * @file dsmil_jadc2_runtime.c
 * @brief DSMIL JADC2 & 5G/MEC Runtime Support (v1.5)
 *
 * Runtime support for Joint All-Domain Command & Control (JADC2) operations
 * over 5G Multi-Access Edge Computing (MEC) networks.
 *
 * Features:
 * - JADC2 transport layer (sensor→C2→shooter pipeline)
 * - 5G/MEC node availability checking
 * - Priority-based message routing
 * - Blue Force Tracker (BFT) integration
 * - Resilient communications (BLOS fallback)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

// JADC2 transport priorities
#define JADC2_PRI_ROUTINE    0    // 0-63: Routine
#define JADC2_PRI_PRIORITY   64   // 64-127: Priority
#define JADC2_PRI_IMMEDIATE  128  // 128-191: Immediate
#define JADC2_PRI_FLASH      192  // 192-255: Flash

// JADC2 domains
typedef enum {
    JADC2_DOMAIN_AIR,
    JADC2_DOMAIN_LAND,
    JADC2_DOMAIN_SEA,
    JADC2_DOMAIN_SPACE,
    JADC2_DOMAIN_CYBER
} jadc2_domain_t;

// BFT (Blue Force Tracker) position
typedef struct {
    double latitude;
    double longitude;
    double altitude;
    uint64_t timestamp_ns;
    char unit_id[64];
} dsmil_bft_position_t;

// JADC2 context (global state)
static struct {
    bool initialized;
    FILE *transport_log;
    uint64_t messages_sent;
    uint64_t messages_received;
    bool mec_available;
    char unit_id[64];
    uint8_t crypto_key[32];
} g_jadc2_ctx = {0};

/**
 * @brief Initialize JADC2 transport layer
 *
 * @param profile_name JADC2 profile (sensor_fusion, c2_processing, etc.)
 * @return 0 on success, negative on error
 */
int dsmil_jadc2_init(const char *profile_name) {
    if (g_jadc2_ctx.initialized) {
        return 0;
    }

    // Open transport log
    const char *log_path = getenv("DSMIL_JADC2_LOG");
    if (!log_path) {
        log_path = "/var/log/dsmil/jadc2_transport.log";
    }

    g_jadc2_ctx.transport_log = fopen(log_path, "a");
    if (!g_jadc2_ctx.transport_log) {
        g_jadc2_ctx.transport_log = stderr;
    }

    // Check for 5G/MEC availability (simplified)
    const char *mec_enable = getenv("DSMIL_5G_MEC_ENABLE");
    g_jadc2_ctx.mec_available = (mec_enable && strcmp(mec_enable, "1") == 0);

    g_jadc2_ctx.initialized = true;
    g_jadc2_ctx.messages_sent = 0;
    g_jadc2_ctx.messages_received = 0;

    snprintf(g_jadc2_ctx.unit_id, sizeof(g_jadc2_ctx.unit_id),
             "UNIT_%d", getpid());

    fprintf(g_jadc2_ctx.transport_log,
            "[INIT] JADC2 transport initialized, profile=%s, mec=%s, unit=%s\n",
            profile_name,
            g_jadc2_ctx.mec_available ? "available" : "unavailable",
            g_jadc2_ctx.unit_id);
    fflush(g_jadc2_ctx.transport_log);

    return 0;
}

/**
 * @brief Send data via JADC2 transport (sensor→C2→shooter pipeline)
 *
 * @param data Message data
 * @param length Message length
 * @param priority Priority level (0-255)
 * @param destination_domain Target domain (air, land, sea, space, cyber)
 * @return 0 on success, negative on error
 */
int dsmil_jadc2_send(const void *data,
                      size_t length,
                      uint8_t priority,
                      const char *destination_domain) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    // Get timestamp
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                             (uint64_t)ts.tv_nsec;

    // Priority classification
    const char *pri_str = "ROUTINE";
    if (priority >= JADC2_PRI_FLASH)
        pri_str = "FLASH";
    else if (priority >= JADC2_PRI_IMMEDIATE)
        pri_str = "IMMEDIATE";
    else if (priority >= JADC2_PRI_PRIORITY)
        pri_str = "PRIORITY";

    // Log transmission
    fprintf(g_jadc2_ctx.transport_log,
            "[SEND] ts=%lu domain=%s priority=%s(%d) bytes=%zu unit=%s\n",
            timestamp_ns,
            destination_domain,
            pri_str,
            priority,
            length,
            g_jadc2_ctx.unit_id);
    fflush(g_jadc2_ctx.transport_log);

    g_jadc2_ctx.messages_sent++;

    // In production: actual network transmission via 5G/MEC
    // For now: simulated
    (void)data;  // Avoid unused warning

    return 0;
}

/**
 * @brief Check if 5G/MEC edge node is available
 *
 * @return true if MEC available, false otherwise
 */
bool dsmil_5g_edge_available(void) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    return g_jadc2_ctx.mec_available;
}

/**
 * @brief Initialize Blue Force Tracker (BFT) subsystem
 *
 * @param unit_id Unit identifier
 * @param crypto_key AES-256 key for BFT encryption (32 bytes)
 * @return 0 on success, negative on error
 */
int dsmil_bft_init(const char *unit_id, const char *crypto_key) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    snprintf(g_jadc2_ctx.unit_id, sizeof(g_jadc2_ctx.unit_id), "%s", unit_id);

    if (crypto_key) {
        memcpy(g_jadc2_ctx.crypto_key, crypto_key, 32);
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[BFT_INIT] unit=%s\n", unit_id);
    fflush(g_jadc2_ctx.transport_log);

    return 0;
}

/**
 * @brief Send BFT position update
 *
 * @param lat Latitude
 * @param lon Longitude
 * @param alt Altitude (meters)
 * @param timestamp_ns Timestamp (nanoseconds since epoch)
 * @return 0 on success, negative on error
 */
int dsmil_bft_send_position(double lat, double lon, double alt,
                              uint64_t timestamp_ns) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[BFT_POS] unit=%s lat=%.6f lon=%.6f alt=%.1f ts=%lu\n",
            g_jadc2_ctx.unit_id, lat, lon, alt, timestamp_ns);
    fflush(g_jadc2_ctx.transport_log);

    // In production: encrypted BFT transmission
    // Encrypt with AES-256 using g_jadc2_ctx.crypto_key
    // Send via BFT-2 protocol

    return 0;
}

/**
 * @brief Receive friendly positions from BFT network
 *
 * @param positions Output array of positions
 * @param max_count Maximum number of positions to receive
 * @return Number of positions received, negative on error
 */
int dsmil_bft_recv_positions(dsmil_bft_position_t *positions,
                               size_t max_count) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    // In production: receive from BFT network
    // For now: return 0 (no positions)
    (void)positions;
    (void)max_count;

    return 0;
}

/**
 * @brief Initialize resilient transport with BLOS fallback
 *
 * @param primary Primary transport ("5g", "link16", "satcom", "muos")
 * @param secondary Fallback transport
 * @return 0 on success, negative on error
 */
int dsmil_blos_init(const char *primary, const char *secondary) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[BLOS_INIT] primary=%s secondary=%s\n", primary, secondary);
    fflush(g_jadc2_ctx.transport_log);

    return 0;
}

/**
 * @brief Send with automatic fallback if primary link jammed
 *
 * @param data Message data
 * @param length Message length
 * @return 0 on success, negative on error
 */
int dsmil_resilient_send(const void *data, size_t length) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    // Check if primary link (5G) available
    if (g_jadc2_ctx.mec_available) {
        fprintf(g_jadc2_ctx.transport_log,
                "[RESILIENT] Using primary link (5G), bytes=%zu\n", length);
        fflush(g_jadc2_ctx.transport_log);

        // Send via 5G
        return dsmil_jadc2_send(data, length, JADC2_PRI_PRIORITY, "land");
    } else {
        fprintf(g_jadc2_ctx.transport_log,
                "[RESILIENT] Primary jammed, fallback to SATCOM, bytes=%zu\n", length);
        fflush(g_jadc2_ctx.transport_log);

        // Fallback to SATCOM (high latency but reliable)
        // In production: adjust timeouts for 100-500ms SATCOM latency
        return 0;
    }
}

/**
 * @brief Activate EMCON (emission control) mode
 *
 * @param level EMCON level (1-4, higher = more restrictive)
 */
void dsmil_emcon_activate(uint8_t level) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[EMCON] Activated level %d (1=normal, 4=RF silent)\n", level);
    fflush(g_jadc2_ctx.transport_log);

    // In production:
    // - Level 2: Suppress non-essential transmissions
    // - Level 3: Batch and delay all transmissions
    // - Level 4: No transmissions except emergency
}

/**
 * @brief Send data in EMCON mode (batched, delayed)
 *
 * @param data Message data
 * @param length Message length
 * @return 0 on success, negative on error
 */
int dsmil_emcon_send(const void *data, size_t length) {
    if (!g_jadc2_ctx.initialized) {
        dsmil_jadc2_init("default");
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[EMCON_SEND] Batching message, bytes=%zu\n", length);
    fflush(g_jadc2_ctx.transport_log);

    // In production: batch messages, delay transmission
    // Add jitter to avoid pattern detection
    (void)data;

    return 0;
}

/**
 * @brief Get timestamp in nanoseconds
 *
 * @return Timestamp (ns since epoch)
 */
uint64_t dsmil_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * @brief Shutdown JADC2 subsystem
 */
void dsmil_jadc2_shutdown(void) {
    if (!g_jadc2_ctx.initialized) {
        return;
    }

    fprintf(g_jadc2_ctx.transport_log,
            "[SHUTDOWN] Messages sent: %lu, received: %lu\n",
            g_jadc2_ctx.messages_sent,
            g_jadc2_ctx.messages_received);

    if (g_jadc2_ctx.transport_log != stderr) {
        fclose(g_jadc2_ctx.transport_log);
    }

    g_jadc2_ctx.initialized = false;
}
