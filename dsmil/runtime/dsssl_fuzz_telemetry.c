/**
 * @file dsssl_fuzz_telemetry.c
 * @brief DSSSL Fuzzing & Telemetry Runtime Implementation
 *
 * Implements telemetry collection, ring buffer management, and budget
 * enforcement for DSSSL fuzzing builds.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsssl_fuzz_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>
#include <pthread.h>
#include <yaml.h>

#define DEFAULT_RING_BUFFER_SIZE 65536
#define MAX_EVENT_SIZE 256

// Ring buffer for events
static dsssl_telemetry_event_t *ring_buffer = NULL;
static size_t ring_buffer_size = 0;
static _Atomic size_t ring_buffer_head = 0;
static _Atomic size_t ring_buffer_tail = 0;
static _Atomic int telemetry_enabled = 0;

// Thread-local context ID
static __thread uint64_t thread_context_id = 0;

// Budget configuration
typedef struct {
    char op_name[64];
    uint32_t max_branches;
    uint32_t max_loads;
    uint32_t max_stores;
    uint64_t max_cycles;
} crypto_budget_t;

static crypto_budget_t *crypto_budgets = NULL;
static size_t num_budgets = 0;

/**
 * Get current timestamp in nanoseconds
 */
static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * Get thread ID
 */
static uint32_t get_thread_id(void) {
    return (uint32_t)pthread_self();
}

/**
 * Add event to ring buffer
 */
static void add_event(const dsssl_telemetry_event_t *ev) {
    if (!telemetry_enabled || !ring_buffer) {
        return;
    }

    size_t head = atomic_load(&ring_buffer_head);
    size_t next_head = (head + 1) % ring_buffer_size;
    
    // Check if buffer is full
    if (next_head == atomic_load(&ring_buffer_tail)) {
        // Buffer full - advance tail (drop oldest event)
        atomic_fetch_add(&ring_buffer_tail, 1);
    }

    // Copy event
    ring_buffer[head] = *ev;
    ring_buffer[head].timestamp = get_timestamp_ns();
    ring_buffer[head].thread_id = get_thread_id();
    ring_buffer[head].context_id = thread_context_id;

    // Advance head
    atomic_store(&ring_buffer_head, next_head);
}

int dsssl_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size_param) {
    if (telemetry_enabled) {
        return 0;  // Already initialized
    }

    // Allocate ring buffer
    ring_buffer_size = ring_buffer_size_param > 0 ? ring_buffer_size_param : DEFAULT_RING_BUFFER_SIZE;
    ring_buffer = calloc(ring_buffer_size, sizeof(dsssl_telemetry_event_t));
    if (!ring_buffer) {
        return -1;
    }

    // Load config if provided
    if (config_path) {
        // TODO: Parse YAML config for budgets
        // For now, use defaults
    }

    atomic_store(&telemetry_enabled, 1);
    return 0;
}

void dsssl_fuzz_telemetry_shutdown(void) {
    atomic_store(&telemetry_enabled, 0);
    
    if (ring_buffer) {
        free(ring_buffer);
        ring_buffer = NULL;
    }
    
    if (crypto_budgets) {
        free(crypto_budgets);
        crypto_budgets = NULL;
    }
}

void dsssl_fuzz_set_context(uint64_t context_id) {
    thread_context_id = context_id;
}

uint64_t dsssl_fuzz_get_context(void) {
    return thread_context_id;
}

void dsssl_cov_hit(uint32_t site_id) {
    dsssl_telemetry_event_t ev = {0};
    ev.event_type = DSSSL_EVENT_COVERAGE_HIT;
    ev.data.coverage.site_id = site_id;
    add_event(&ev);
}

void dsssl_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to) {
    dsssl_telemetry_event_t ev = {0};
    ev.event_type = DSSSL_EVENT_STATE_TRANSITION;
    ev.data.state_transition.sm_id = sm_id;
    ev.data.state_transition.state_from = state_from;
    ev.data.state_transition.state_to = state_to;
    add_event(&ev);
}

void dsssl_crypto_metric_begin(const char *op_name) {
    // Begin tracking - implementation would start counters
    (void)op_name;
}

void dsssl_crypto_metric_end(const char *op_name) {
    // End tracking - implementation would record metrics
    (void)op_name;
}

void dsssl_crypto_metric_record(const char *op_name, uint32_t branches,
                                uint32_t loads, uint32_t stores, uint64_t cycles) {
    dsssl_telemetry_event_t ev = {0};
    ev.event_type = DSSSL_EVENT_CRYPTO_METRIC;
    ev.data.crypto_metric.op_name = op_name;
    ev.data.crypto_metric.branches = branches;
    ev.data.crypto_metric.loads = loads;
    ev.data.crypto_metric.stores = stores;
    ev.data.crypto_metric.cycles = cycles;
    add_event(&ev);

    // Check budget
    if (dsssl_crypto_check_budget(op_name, branches, loads, stores, cycles)) {
        dsssl_telemetry_event_t budget_ev = {0};
        budget_ev.event_type = DSSSL_EVENT_BUDGET_VIOLATION;
        budget_ev.data.budget_violation.budget_name = op_name;
        budget_ev.data.budget_violation.actual = branches + loads + stores;
        budget_ev.data.budget_violation.limit = 0;  // Would be set from config
        add_event(&budget_ev);
    }
}

void dsssl_api_misuse_report(const char *api, const char *reason, uint64_t context_id) {
    dsssl_telemetry_event_t ev = {0};
    ev.event_type = DSSSL_EVENT_API_MISUSE;
    ev.data.api_misuse.api = api;
    ev.data.api_misuse.reason = reason;
    ev.data.api_misuse.context_id = context_id;
    add_event(&ev);
}

void dsssl_ticket_event(dsssl_ticket_event_t subtype, uint64_t ticket_id) {
    dsssl_telemetry_event_t ev = {0};
    ev.event_type = DSSSL_EVENT_TICKET_EVENT;
    ev.data.ticket_event.subtype = subtype;
    ev.data.ticket_event.ticket_id = ticket_id;
    add_event(&ev);
}

size_t dsssl_fuzz_get_events(dsssl_telemetry_event_t *events, size_t max_events) {
    if (!ring_buffer || !events) {
        return 0;
    }

    size_t tail = atomic_load(&ring_buffer_tail);
    size_t head = atomic_load(&ring_buffer_head);
    size_t count = 0;

    while (tail != head && count < max_events) {
        events[count] = ring_buffer[tail];
        tail = (tail + 1) % ring_buffer_size;
        count++;
    }

    atomic_store(&ring_buffer_tail, tail);
    return count;
}

int dsssl_fuzz_flush_events(const char *filepath) {
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        return -1;
    }

    dsssl_telemetry_event_t events[1024];
    size_t count = dsssl_fuzz_get_events(events, 1024);
    
    while (count > 0) {
        fwrite(events, sizeof(dsssl_telemetry_event_t), count, fp);
        count = dsssl_fuzz_get_events(events, 1024);
    }

    fclose(fp);
    return 0;
}

void dsssl_fuzz_clear_events(void) {
    atomic_store(&ring_buffer_tail, atomic_load(&ring_buffer_head));
}

int dsssl_crypto_check_budget(const char *op_name, uint32_t branches,
                              uint32_t loads, uint32_t stores, uint64_t cycles) {
    // Check against budgets (simplified - full implementation would look up in config)
    // For now, use hardcoded limits
    if (branches > 10000 || loads > 50000 || stores > 50000) {
        return 1;  // Violation
    }
    return 0;  // Within budget
}
