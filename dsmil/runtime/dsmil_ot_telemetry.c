/**
 * @file dsmil_ot_telemetry.c
 * @brief DSLLVM OT Telemetry Runtime Implementation
 *
 * Async-safe implementation of OT telemetry logging with minimal runtime
 * overhead. Uses ring buffer or simple stderr logging.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>

// Simple ring buffer implementation for async-safe logging
#define RING_BUFFER_SIZE 1024
#define MAX_EVENT_SIZE 512

typedef struct {
    char buffer[MAX_EVENT_SIZE];
    size_t len;
    _Atomic int used;
} ring_buffer_entry_t;

static ring_buffer_entry_t ring_buffer[RING_BUFFER_SIZE];
static _Atomic size_t ring_buffer_head = 0;
static _Atomic int telemetry_enabled = -1;  // -1 = uninitialized, 0 = disabled, 1 = enabled
static _Atomic dsmil_telemetry_level_t telemetry_level = DSMIL_TELEMETRY_LEVEL_NORMAL;
static _Atomic int telemetry_level_initialized = 0;

/**
 * Parse telemetry level from string
 */
static dsmil_telemetry_level_t parse_telemetry_level(const char *str) {
    if (!str) return DSMIL_TELEMETRY_LEVEL_NORMAL;
    
    if (strcmp(str, "off") == 0) return DSMIL_TELEMETRY_LEVEL_OFF;
    if (strcmp(str, "min") == 0) return DSMIL_TELEMETRY_LEVEL_MIN;
    if (strcmp(str, "normal") == 0) return DSMIL_TELEMETRY_LEVEL_NORMAL;
    if (strcmp(str, "debug") == 0) return DSMIL_TELEMETRY_LEVEL_DEBUG;
    if (strcmp(str, "trace") == 0) return DSMIL_TELEMETRY_LEVEL_TRACE;
    
    return DSMIL_TELEMETRY_LEVEL_NORMAL;  // Default
}

/**
 * Get compile-time telemetry level (from weak symbol or module flag)
 */
static dsmil_telemetry_level_t get_compile_time_level(void) {
    // Try to read from weak symbol emitted by pass
    // For now, default to normal
    extern dsmil_telemetry_level_t __start_dsmil_config __attribute__((weak));
    if (&__start_dsmil_config != NULL) {
        // In a full implementation, would read from __start_dsmil_config
        // For now, check environment variable as fallback
    }
    return DSMIL_TELEMETRY_LEVEL_NORMAL;
}

/**
 * Initialize telemetry level (combines compile-time and runtime)
 */
static void init_telemetry_level(void) {
    if (atomic_load(&telemetry_level_initialized)) {
        return;
    }
    
    dsmil_telemetry_level_t compile_level = get_compile_time_level();
    dsmil_telemetry_level_t runtime_level = DSMIL_TELEMETRY_LEVEL_NORMAL;
    
    // Check runtime override
    const char *env_level = getenv("DSMIL_TELEMETRY_LEVEL");
    if (env_level) {
        runtime_level = parse_telemetry_level(env_level);
    }
    
    // Enforce lattice: take maximum (more verbose)
    dsmil_telemetry_level_t final_level = compile_level > runtime_level ? compile_level : runtime_level;
    
    // Mission profile overrides (check DSMIL_MISSION_PROFILE)
    const char *mission_profile = getenv("DSMIL_MISSION_PROFILE");
    if (mission_profile) {
        // Production profiles force minimum levels unless CLI demanded stricter
        if (strcmp(mission_profile, "ics_prod") == 0 || 
            strcmp(mission_profile, "border_ops") == 0) {
            if (final_level < DSMIL_TELEMETRY_LEVEL_MIN) {
                final_level = DSMIL_TELEMETRY_LEVEL_MIN;
            }
        }
    }
    
    atomic_store(&telemetry_level, final_level);
    atomic_store(&telemetry_level_initialized, 1);
}

/**
 * Check if telemetry is enabled (lazy initialization)
 */
static int check_telemetry_enabled(void) {
    int enabled = atomic_load(&telemetry_enabled);
    if (enabled == -1) {
        // Check environment variable
        const char *env = getenv("DSMIL_OT_TELEMETRY");
        if (env && (env[0] == '0' || env[0] == 'f' || env[0] == 'F')) {
            enabled = 0;
        } else {
            // Default: enabled in production
            enabled = 1;
        }
        atomic_store(&telemetry_enabled, enabled);
    }
    return enabled;
}

/**
 * Format event as JSON line
 */
static size_t format_event_json(const dsmil_telemetry_event_t *ev, char *buf, size_t buf_size) {
    const char *event_type_str = "unknown";
    switch (ev->event_type) {
        case DSMIL_TELEMETRY_OT_PATH_ENTRY: event_type_str = "ot_path_entry"; break;
        case DSMIL_TELEMETRY_OT_PATH_EXIT:  event_type_str = "ot_path_exit"; break;
        case DSMIL_TELEMETRY_SES_INTENT:    event_type_str = "ses_intent"; break;
        case DSMIL_TELEMETRY_SES_ACCEPT:    event_type_str = "ses_accept"; break;
        case DSMIL_TELEMETRY_SES_REJECT:    event_type_str = "ses_reject"; break;
        case DSMIL_TELEMETRY_INVARIANT_HIT: event_type_str = "invariant_hit"; break;
        case DSMIL_TELEMETRY_INVARIANT_FAIL: event_type_str = "invariant_fail"; break;
        case DSMIL_TELEMETRY_SS7_MSG_RX: event_type_str = "ss7_msg_rx"; break;
        case DSMIL_TELEMETRY_SS7_MSG_TX: event_type_str = "ss7_msg_tx"; break;
        case DSMIL_TELEMETRY_SIGTRAN_MSG_RX: event_type_str = "sigtran_msg_rx"; break;
        case DSMIL_TELEMETRY_SIGTRAN_MSG_TX: event_type_str = "sigtran_msg_tx"; break;
        case DSMIL_TELEMETRY_SIG_ANOMALY: event_type_str = "sig_anomaly"; break;
    }

    // Get timestamp
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

    int written = snprintf(buf, buf_size,
        "{\"type\":\"%s\",\"ts\":%llu,\"module\":\"%s\",\"func\":\"%s\","
        "\"file\":\"%s\",\"line\":%u,\"layer\":%u,\"device\":%u,"
        "\"stage\":\"%s\",\"profile\":\"%s\",\"tier\":%u,"
        "\"build_id\":%llu,\"provenance_id\":%llu",
        event_type_str, (unsigned long long)timestamp_ns,
        ev->module_id ? ev->module_id : "unknown",
        ev->func_id ? ev->func_id : "unknown",
        ev->file ? ev->file : "unknown",
        ev->line,
        ev->layer,
        ev->device,
        ev->stage ? ev->stage : "",
        ev->mission_profile ? ev->mission_profile : "",
        ev->authority_tier,
        (unsigned long long)ev->build_id,
        (unsigned long long)ev->provenance_id);

    // Add signal data if present
    if (ev->signal_name) {
        written += snprintf(buf + written, buf_size - written,
            ",\"signal\":\"%s\",\"value\":%.6f,\"min\":%.6f,\"max\":%.6f",
            ev->signal_name, ev->signal_value, ev->signal_min, ev->signal_max);
    }

    // Add generic annotation fields if present (backward compatible)
    if (ev->category) {
        written += snprintf(buf + written, buf_size - written,
            ",\"category\":\"%s\"", ev->category);
    }
    if (ev->op) {
        written += snprintf(buf + written, buf_size - written,
            ",\"op\":\"%s\"", ev->op);
    }
    if (ev->status_code != 0) {
        written += snprintf(buf + written, buf_size - written,
            ",\"status_code\":%d", ev->status_code);
    }
    if (ev->resource) {
        written += snprintf(buf + written, buf_size - written,
            ",\"resource\":\"%s\"", ev->resource);
    }
    if (ev->error_msg) {
        written += snprintf(buf + written, buf_size - written,
            ",\"error_msg\":\"%s\"", ev->error_msg);
    }
    if (ev->elapsed_ns > 0) {
        written += snprintf(buf + written, buf_size - written,
            ",\"elapsed_ns\":%llu", (unsigned long long)ev->elapsed_ns);
    }

    written += snprintf(buf + written, buf_size - written, "}\n");
    return (size_t)written;
}

/**
 * Log event to stderr (simple implementation)
 */
static void log_to_stderr(const dsmil_telemetry_event_t *ev) {
    char buf[MAX_EVENT_SIZE];
    size_t len = format_event_json(ev, buf, sizeof(buf));
    if (len > 0 && len < sizeof(buf)) {
        fwrite(buf, 1, len, stderr);
        fflush(stderr);
    }
}

/**
 * Log event using ring buffer (async-safe)
 */
static void log_to_ring_buffer(const dsmil_telemetry_event_t *ev) {
    if (!check_telemetry_enabled()) {
        return;
    }

    char buf[MAX_EVENT_SIZE];
    size_t len = format_event_json(ev, buf, sizeof(buf));
    if (len == 0 || len >= sizeof(buf)) {
        return;  // Event too large or formatting failed
    }

    // Get next ring buffer slot
    size_t idx = atomic_fetch_add(&ring_buffer_head, 1) % RING_BUFFER_SIZE;
    ring_buffer_entry_t *entry = &ring_buffer[idx];

    // Wait for slot to be free (simple spin-wait, async-safe)
    int expected = 0;
    while (!atomic_compare_exchange_weak(&entry->used, &expected, 1)) {
        expected = 0;
    }

    // Copy event data
    memcpy(entry->buffer, buf, len);
    entry->len = len;

    // Mark as ready (consumer will reset 'used' flag)
    // For now, we'll just write to stderr directly
    // In a full implementation, a background thread would consume the ring buffer
    fwrite(entry->buffer, 1, entry->len, stderr);
    fflush(stderr);

    atomic_store(&entry->used, 0);
}

void dsmil_telemetry_event(const dsmil_telemetry_event_t *ev) {
    if (!ev || !check_telemetry_enabled()) {
        return;
    }

    // Initialize telemetry level if needed
    init_telemetry_level();

    // Check level gating
    if (!dsmil_telemetry_level_allows(ev->event_type, ev->category)) {
        return;  // Filtered by level
    }

    // Simple implementation: write directly to stderr
    // In production, could use ring buffer + background thread
    log_to_stderr(ev);
}

void dsmil_telemetry_safety_signal_update(const dsmil_telemetry_event_t *ev) {
    if (!ev || !check_telemetry_enabled()) {
        return;
    }

    if (!ev->signal_name) {
        return;  // Invalid event
    }

    // Use same logging path as regular events
    log_to_stderr(ev);
}

int dsmil_ot_telemetry_init(void) {
    // Force re-evaluation of the environment on every init so tests that toggle
    // DSMIL_OT_TELEMETRY between runs get a fresh read.
    atomic_store(&telemetry_enabled, -1);
    check_telemetry_enabled();
    return 0;
}

void dsmil_ot_telemetry_shutdown(void) {
    // Flush any pending events
    fflush(stderr);
    atomic_store(&telemetry_enabled, 0);
    atomic_store(&telemetry_level_initialized, 0);  // Reset for reinit
}

int dsmil_ot_telemetry_is_enabled(void) {
    return check_telemetry_enabled();
}

dsmil_telemetry_level_t dsmil_telemetry_get_level(void) {
    init_telemetry_level();
    return atomic_load(&telemetry_level);
}

int dsmil_telemetry_level_allows(dsmil_telemetry_event_type_t event_type,
                                  const char *category) {
    dsmil_telemetry_level_t level = dsmil_telemetry_get_level();
    
    // Off level: no telemetry
    if (level == DSMIL_TELEMETRY_LEVEL_OFF) {
        return 0;
    }
    
    // Min level: only safety-critical (OT events, errors, panics)
    if (level == DSMIL_TELEMETRY_LEVEL_MIN) {
        return (event_type <= DSMIL_TELEMETRY_INVARIANT_FAIL) ||
               (event_type == DSMIL_TELEMETRY_ERROR) ||
               (event_type == DSMIL_TELEMETRY_PANIC);
    }
    
    // Normal level: entry probes for annotated functions
    // (all OT events, generic annotation events 30-36)
    if (level == DSMIL_TELEMETRY_LEVEL_NORMAL) {
        return event_type >= DSMIL_TELEMETRY_OT_PATH_ENTRY &&
               event_type <= DSMIL_TELEMETRY_PANIC;
    }
    
    // Debug level: entry + exit + timing
    // (all events, but exit events only at debug+)
    if (level == DSMIL_TELEMETRY_LEVEL_DEBUG) {
        return 1;  // All events allowed
    }
    
    // Trace level: everything including sampling
    return 1;  // All events allowed
}
