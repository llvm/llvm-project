/**
 * @file dsmil_fuzz_telemetry.c
 * @brief DSLLVM General-Purpose Fuzzing & Telemetry Runtime Implementation
 *
 * General-purpose telemetry runtime for any fuzzing target.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_fuzz_telemetry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>
#include <pthread.h>

#define DEFAULT_RING_BUFFER_SIZE 65536
#define MAX_EVENT_SIZE 256

// Ring buffer for events
static dsmil_fuzz_telemetry_event_t *ring_buffer = NULL;
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
} operation_budget_t;

static operation_budget_t *operation_budgets = NULL;
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
static void add_event(const dsmil_fuzz_telemetry_event_t *ev) {
    if (!telemetry_enabled || !ring_buffer) {
        return;
    }

    size_t head = atomic_load(&ring_buffer_head);
    size_t next_head = (head + 1) % ring_buffer_size;
    
    if (next_head == atomic_load(&ring_buffer_tail)) {
        atomic_fetch_add(&ring_buffer_tail, 1);
    }

    ring_buffer[head] = *ev;
    ring_buffer[head].timestamp = get_timestamp_ns();
    ring_buffer[head].thread_id = get_thread_id();
    ring_buffer[head].context_id = thread_context_id;

    atomic_store(&ring_buffer_head, next_head);
}

int dsmil_fuzz_telemetry_init(const char *config_path, size_t ring_buffer_size_param) {
    if (telemetry_enabled) {
        return 0;
    }

    ring_buffer_size = ring_buffer_size_param > 0 ? ring_buffer_size_param : DEFAULT_RING_BUFFER_SIZE;
    ring_buffer = calloc(ring_buffer_size, sizeof(dsmil_fuzz_telemetry_event_t));
    if (!ring_buffer) {
        return -1;
    }

    // TODO: Load config if provided

    atomic_store(&telemetry_enabled, 1);
    return 0;
}

void dsmil_fuzz_telemetry_shutdown(void) {
    atomic_store(&telemetry_enabled, 0);
    
    if (ring_buffer) {
        free(ring_buffer);
        ring_buffer = NULL;
    }
    
    if (operation_budgets) {
        free(operation_budgets);
        operation_budgets = NULL;
    }
}

void dsmil_fuzz_set_context(uint64_t context_id) {
    thread_context_id = context_id;
}

uint64_t dsmil_fuzz_get_context(void) {
    return thread_context_id;
}

void dsmil_fuzz_cov_hit(uint32_t site_id) {
    dsmil_fuzz_telemetry_event_t ev = {0};
    ev.event_type = DSMIL_FUZZ_EVENT_COVERAGE_HIT;
    ev.data.coverage.site_id = site_id;
    add_event(&ev);
}

void dsmil_fuzz_state_transition(uint16_t sm_id, uint16_t state_from, uint16_t state_to) {
    dsmil_fuzz_telemetry_event_t ev = {0};
    ev.event_type = DSMIL_FUZZ_EVENT_STATE_TRANSITION;
    ev.data.state_transition.sm_id = sm_id;
    ev.data.state_transition.state_from = state_from;
    ev.data.state_transition.state_to = state_to;
    add_event(&ev);
}

void dsmil_fuzz_metric_begin(const char *op_name) {
    (void)op_name;
    // Begin tracking
}

void dsmil_fuzz_metric_end(const char *op_name) {
    (void)op_name;
    // End tracking
}

void dsmil_fuzz_metric_record(const char *op_name, uint32_t branches,
                              uint32_t loads, uint32_t stores, uint64_t cycles) {
    dsmil_fuzz_telemetry_event_t ev = {0};
    ev.event_type = DSMIL_FUZZ_EVENT_METRIC;
    ev.data.metric.op_name = op_name;
    ev.data.metric.branches = branches;
    ev.data.metric.loads = loads;
    ev.data.metric.stores = stores;
    ev.data.metric.cycles = cycles;
    add_event(&ev);
}

void dsmil_fuzz_api_misuse_report(const char *api, const char *reason, uint64_t context_id) {
    dsmil_fuzz_telemetry_event_t ev = {0};
    ev.event_type = DSMIL_FUZZ_EVENT_API_MISUSE;
    ev.data.api_misuse.api = api;
    ev.data.api_misuse.reason = reason;
    ev.data.api_misuse.context_id = context_id;
    add_event(&ev);
}

void dsmil_fuzz_state_event(dsmil_state_event_t subtype, uint64_t state_id) {
    dsmil_fuzz_telemetry_event_t ev = {0};
    ev.event_type = DSMIL_FUZZ_EVENT_STATE_EVENT;
    ev.data.state_event.subtype = subtype;
    ev.data.state_event.state_id = state_id;
    add_event(&ev);
}

size_t dsmil_fuzz_get_events(dsmil_fuzz_telemetry_event_t *events, size_t max_events) {
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

int dsmil_fuzz_flush_events(const char *filepath) {
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        return -1;
    }

    dsmil_fuzz_telemetry_event_t events[1024];
    size_t count = dsmil_fuzz_get_events(events, 1024);
    
    while (count > 0) {
        fwrite(events, sizeof(dsmil_fuzz_telemetry_event_t), count, fp);
        count = dsmil_fuzz_get_events(events, 1024);
    }

    fclose(fp);
    return 0;
}

void dsmil_fuzz_clear_events(void) {
    atomic_store(&ring_buffer_tail, atomic_load(&ring_buffer_head));
}

int dsmil_fuzz_check_budget(const char *op_name, uint32_t branches,
                           uint32_t loads, uint32_t stores, uint64_t cycles) {
    // Check against budgets (simplified)
    if (branches > 10000 || loads > 50000 || stores > 50000) {
        return 1;  // Violation
    }
    return 0;  // Within budget
}
