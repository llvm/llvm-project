/**
 * @file dsmil_radio_runtime.c
 * @brief DSMIL Tactical Radio Multi-Protocol Runtime (v1.5.1)
 *
 * Multi-protocol tactical radio bridging runtime, inspired by TraX.
 * Supports Link-16, SATCOM, MUOS, SINCGARS, and EPLRS with unified API.
 *
 * Protocol Specifications:
 * - Link-16: J-series messages, 16/31/51/75 bits per word
 * - SATCOM: Various bands (UHF, SHF, EHF), FEC encoding
 * - MUOS: 3G-based WCDMA, 5 kHz channels
 * - SINCGARS: Frequency hopping, 25 kHz channels
 * - EPLRS: Position location reporting, mesh network
 *
 * Features:
 * - Protocol-specific framing and error correction
 * - Automatic protocol selection based on availability
 * - Jamming detection and protocol switching
 * - Unified send/receive API across all protocols
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// Radio protocol types
typedef enum {
    DSMIL_RADIO_LINK16 = 0,
    DSMIL_RADIO_SATCOM = 1,
    DSMIL_RADIO_MUOS = 2,
    DSMIL_RADIO_SINCGARS = 3,
    DSMIL_RADIO_EPLRS = 4
} dsmil_radio_protocol_t;

// Protocol availability status
typedef struct {
    bool link16_available;
    bool satcom_available;
    bool muos_available;
    bool sincgars_available;
    bool eplrs_available;
    dsmil_radio_protocol_t primary_protocol;
} dsmil_radio_status_t;

// Global radio context
static struct {
    bool initialized;
    FILE *radio_log;
    dsmil_radio_status_t status;
    uint64_t messages_sent[5];  // Per-protocol counters
    uint64_t messages_received[5];
    uint64_t jamming_detected[5];
} g_radio_ctx = {0};

/**
 * @brief Initialize radio bridging subsystem
 *
 * @param primary_protocol Preferred primary protocol
 * @return 0 on success, negative on error
 */
int dsmil_radio_init(dsmil_radio_protocol_t primary_protocol) {
    if (g_radio_ctx.initialized) {
        return 0;
    }

    // Open radio log
    const char *log_path = getenv("DSMIL_RADIO_LOG");
    if (!log_path) {
        log_path = "/var/log/dsmil/radio_bridge.log";
    }

    g_radio_ctx.radio_log = fopen(log_path, "a");
    if (!g_radio_ctx.radio_log) {
        g_radio_ctx.radio_log = stderr;
    }

    // Initialize protocol availability (simplified - production would probe hardware)
    g_radio_ctx.status.link16_available = true;
    g_radio_ctx.status.satcom_available = true;
    g_radio_ctx.status.muos_available = true;
    g_radio_ctx.status.sincgars_available = true;
    g_radio_ctx.status.eplrs_available = true;
    g_radio_ctx.status.primary_protocol = primary_protocol;

    g_radio_ctx.initialized = true;

    fprintf(g_radio_ctx.radio_log,
            "[RADIO_INIT] Primary: %d, Link-16=%d SATCOM=%d MUOS=%d SINCGARS=%d EPLRS=%d\n",
            primary_protocol,
            g_radio_ctx.status.link16_available,
            g_radio_ctx.status.satcom_available,
            g_radio_ctx.status.muos_available,
            g_radio_ctx.status.sincgars_available,
            g_radio_ctx.status.eplrs_available);
    fflush(g_radio_ctx.radio_log);

    return 0;
}

/**
 * @brief Frame message for Link-16 (J-series)
 *
 * Link-16 uses J-series messages with specific formatting.
 * Messages are 75-bit words with error correction.
 */
int dsmil_radio_frame_link16(const uint8_t *data, size_t length,
                               uint8_t *output) {
    // Link-16 framing: add J-series header
    // Production would implement actual Link-16 J-series formatting
    fprintf(g_radio_ctx.radio_log,
            "[LINK16_FRAME] Framing %zu bytes as J-series message\n", length);
    fflush(g_radio_ctx.radio_log);

    // Simplified: copy with header
    output[0] = 0x4A;  // 'J' for J-series
    output[1] = (uint8_t)(length & 0xFF);
    memcpy(output + 2, data, length);

    return (int)(length + 2);
}

/**
 * @brief Frame message for SATCOM
 *
 * SATCOM requires FEC (Forward Error Correction) for lossy satellite links.
 */
int dsmil_radio_frame_satcom(const uint8_t *data, size_t length,
                               uint8_t *output) {
    // SATCOM framing: add FEC encoding
    fprintf(g_radio_ctx.radio_log,
            "[SATCOM_FRAME] Framing %zu bytes with FEC\n", length);
    fflush(g_radio_ctx.radio_log);

    // Simplified: add FEC header and parity
    output[0] = 0xFE;  // FEC marker
    output[1] = 0xC0;  // FEC code
    memcpy(output + 2, data, length);

    // Add simple parity (production would use Reed-Solomon or similar)
    uint8_t parity = 0;
    for (size_t i = 0; i < length; i++) {
        parity ^= data[i];
    }
    output[length + 2] = parity;

    return (int)(length + 3);
}

/**
 * @brief Frame message for MUOS (3G-based WCDMA)
 */
int dsmil_radio_frame_muos(const uint8_t *data, size_t length,
                            uint8_t *output) {
    fprintf(g_radio_ctx.radio_log,
            "[MUOS_FRAME] Framing %zu bytes for WCDMA\n", length);
    fflush(g_radio_ctx.radio_log);

    // MUOS uses 3G-like framing
    output[0] = 0x03;  // Simplified marker
    memcpy(output + 1, data, length);

    return (int)(length + 1);
}

/**
 * @brief Frame message for SINCGARS (frequency hopping)
 */
int dsmil_radio_frame_sincgars(const uint8_t *data, size_t length,
                                 uint8_t *output) {
    fprintf(g_radio_ctx.radio_log,
            "[SINCGARS_FRAME] Framing %zu bytes for freq hopping\n", length);
    fflush(g_radio_ctx.radio_log);

    // SINCGARS: add hop pattern indicator
    output[0] = 0x25;  // 25 kHz channel indicator
    memcpy(output + 1, data, length);

    return (int)(length + 1);
}

/**
 * @brief Frame message for EPLRS (position location reporting)
 */
int dsmil_radio_frame_eplrs(const uint8_t *data, size_t length,
                              uint8_t *output) {
    fprintf(g_radio_ctx.radio_log,
            "[EPLRS_FRAME] Framing %zu bytes for EPLRS mesh\n", length);
    fflush(g_radio_ctx.radio_log);

    // EPLRS: mesh network framing
    output[0] = 0x0E;  // EPLRS marker
    memcpy(output + 1, data, length);

    return (int)(length + 1);
}

/**
 * @brief Unified radio bridge send function
 *
 * Automatically selects best available protocol and sends message.
 *
 * @param protocol Preferred protocol (NULL for automatic selection)
 * @param data Message data
 * @param length Message length
 * @return 0 on success, negative on error
 */
int dsmil_radio_bridge_send(const char *protocol, const uint8_t *data,
                              size_t length) {
    if (!g_radio_ctx.initialized) {
        dsmil_radio_init(DSMIL_RADIO_LINK16);
    }

    // Determine which protocol to use
    dsmil_radio_protocol_t selected_proto = g_radio_ctx.status.primary_protocol;

    if (protocol) {
        // User specified protocol
        if (strcmp(protocol, "link16") == 0)
            selected_proto = DSMIL_RADIO_LINK16;
        else if (strcmp(protocol, "satcom") == 0)
            selected_proto = DSMIL_RADIO_SATCOM;
        else if (strcmp(protocol, "muos") == 0)
            selected_proto = DSMIL_RADIO_MUOS;
        else if (strcmp(protocol, "sincgars") == 0)
            selected_proto = DSMIL_RADIO_SINCGARS;
        else if (strcmp(protocol, "eplrs") == 0)
            selected_proto = DSMIL_RADIO_EPLRS;
    }

    // Check availability and fallback if necessary
    bool available = false;
    switch (selected_proto) {
        case DSMIL_RADIO_LINK16:
            available = g_radio_ctx.status.link16_available;
            break;
        case DSMIL_RADIO_SATCOM:
            available = g_radio_ctx.status.satcom_available;
            break;
        case DSMIL_RADIO_MUOS:
            available = g_radio_ctx.status.muos_available;
            break;
        case DSMIL_RADIO_SINCGARS:
            available = g_radio_ctx.status.sincgars_available;
            break;
        case DSMIL_RADIO_EPLRS:
            available = g_radio_ctx.status.eplrs_available;
            break;
    }

    if (!available) {
        fprintf(g_radio_ctx.radio_log,
                "[RADIO_BRIDGE] Protocol %d unavailable, trying fallback\n",
                selected_proto);
        fflush(g_radio_ctx.radio_log);

        // Try SATCOM as fallback (usually most reliable)
        if (g_radio_ctx.status.satcom_available) {
            selected_proto = DSMIL_RADIO_SATCOM;
        } else {
            return -1;  // No available protocol
        }
    }

    // Frame message for selected protocol
    uint8_t framed[4096];
    int framed_len = 0;

    switch (selected_proto) {
        case DSMIL_RADIO_LINK16:
            framed_len = dsmil_radio_frame_link16(data, length, framed);
            break;
        case DSMIL_RADIO_SATCOM:
            framed_len = dsmil_radio_frame_satcom(data, length, framed);
            break;
        case DSMIL_RADIO_MUOS:
            framed_len = dsmil_radio_frame_muos(data, length, framed);
            break;
        case DSMIL_RADIO_SINCGARS:
            framed_len = dsmil_radio_frame_sincgars(data, length, framed);
            break;
        case DSMIL_RADIO_EPLRS:
            framed_len = dsmil_radio_frame_eplrs(data, length, framed);
            break;
    }

    if (framed_len < 0) {
        return -1;
    }

    // Send via selected protocol (production would use actual radio hardware)
    fprintf(g_radio_ctx.radio_log,
            "[RADIO_BRIDGE_TX] protocol=%d bytes_original=%zu bytes_framed=%d\n",
            selected_proto, length, framed_len);
    fflush(g_radio_ctx.radio_log);

    g_radio_ctx.messages_sent[selected_proto]++;

    return 0;
}

/**
 * @brief Detect jamming on protocol
 *
 * @param protocol Protocol to check
 * @return true if jamming detected, false otherwise
 */
bool dsmil_radio_detect_jamming(dsmil_radio_protocol_t protocol) {
    // Production would analyze signal strength, bit error rate, etc.
    // For now: simulated (check environment variable)

    const char *jam_env = getenv("DSMIL_RADIO_JAMMING");
    if (jam_env) {
        int jammed_proto = atoi(jam_env);
        if (jammed_proto == (int)protocol) {
            g_radio_ctx.jamming_detected[protocol]++;
            fprintf(g_radio_ctx.radio_log,
                    "[RADIO_JAMMING] Protocol %d jammed!\n", protocol);
            fflush(g_radio_ctx.radio_log);
            return true;
        }
    }

    return false;
}

/**
 * @brief Get radio status
 */
void dsmil_radio_get_status(dsmil_radio_status_t *status) {
    if (!g_radio_ctx.initialized) {
        memset(status, 0, sizeof(*status));
        return;
    }

    // Update availability based on jamming detection
    g_radio_ctx.status.link16_available = !dsmil_radio_detect_jamming(DSMIL_RADIO_LINK16);
    g_radio_ctx.status.satcom_available = !dsmil_radio_detect_jamming(DSMIL_RADIO_SATCOM);

    *status = g_radio_ctx.status;
}

/**
 * @brief Get radio statistics
 */
void dsmil_radio_get_stats(uint64_t *sent, uint64_t *received, uint64_t *jamming) {
    if (!g_radio_ctx.initialized) {
        return;
    }

    memcpy(sent, g_radio_ctx.messages_sent, sizeof(g_radio_ctx.messages_sent));
    memcpy(received, g_radio_ctx.messages_received, sizeof(g_radio_ctx.messages_received));
    memcpy(jamming, g_radio_ctx.jamming_detected, sizeof(g_radio_ctx.jamming_detected));
}

/**
 * @brief Shutdown radio subsystem
 */
void dsmil_radio_shutdown(void) {
    if (!g_radio_ctx.initialized) {
        return;
    }

    fprintf(g_radio_ctx.radio_log,
            "[RADIO_SHUTDOWN] Link16: %lu SATCOM: %lu MUOS: %lu SINCGARS: %lu EPLRS: %lu\n",
            g_radio_ctx.messages_sent[0],
            g_radio_ctx.messages_sent[1],
            g_radio_ctx.messages_sent[2],
            g_radio_ctx.messages_sent[3],
            g_radio_ctx.messages_sent[4]);

    if (g_radio_ctx.radio_log != stderr) {
        fclose(g_radio_ctx.radio_log);
    }

    g_radio_ctx.initialized = false;
}
