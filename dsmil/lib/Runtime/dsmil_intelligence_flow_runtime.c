/**
 * @file dsmil_intelligence_flow_runtime.c
 * @brief Cross-Layer Intelligence Flow Runtime Implementation
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_intelligence_flow.h"
#include "dsmil_cross_domain_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SUBSCRIPTIONS 256
#define MAX_EVENTS_QUEUE 1024

typedef struct {
    uint8_t layer;
    uint32_t device;
    dsmil_intelligence_type_t intel_type;
    dsmil_intelligence_callback_t callback;
    bool active;
} intelligence_subscription_t;

typedef struct {
    bool initialized;
    intelligence_subscription_t subscriptions[MAX_SUBSCRIPTIONS];
    uint32_t num_subscriptions;
    dsmil_intelligence_event_t event_queue[MAX_EVENTS_QUEUE];
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;
} intelligence_flow_state_t;

static intelligence_flow_state_t g_intel_state = {0};

static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int dsmil_intelligence_flow_init(void) {
    if (g_intel_state.initialized) {
        return 0;
    }
    
    memset(&g_intel_state, 0, sizeof(g_intel_state));
    g_intel_state.initialized = true;
    
    return 0;
}

int dsmil_intelligence_publish(const dsmil_intelligence_event_t *event) {
    if (!g_intel_state.initialized) {
        if (dsmil_intelligence_flow_init() != 0) {
            return -1;
        }
    }
    
    if (!event) {
        return -1;
    }
    
    // Verify upward flow (target layer >= source layer)
    if (event->target_layer < event->source_layer) {
        fprintf(stderr, "ERROR: Invalid downward flow from layer %u to %u\n",
                event->source_layer, event->target_layer);
        return -1;
    }
    
    // Verify clearance
    if (!dsmil_intelligence_verify_clearance(event->source_layer,
                                            event->target_layer,
                                            event->clearance_mask)) {
        fprintf(stderr, "ERROR: Clearance verification failed\n");
        return -1;
    }
    
    // Add timestamp if not set
    dsmil_intelligence_event_t event_copy = *event;
    if (event_copy.timestamp_ns == 0) {
        event_copy.timestamp_ns = get_timestamp_ns();
    }
    
    // Queue event
    if (g_intel_state.queue_count < MAX_EVENTS_QUEUE) {
        uint32_t idx = (g_intel_state.queue_tail + g_intel_state.queue_count) % MAX_EVENTS_QUEUE;
        g_intel_state.event_queue[idx] = event_copy;
        g_intel_state.queue_count++;
    } else {
        fprintf(stderr, "WARNING: Event queue full, dropping event\n");
        return -1;
    }
    
    // Notify subscribers
    for (uint32_t i = 0; i < g_intel_state.num_subscriptions; i++) {
        intelligence_subscription_t *sub = &g_intel_state.subscriptions[i];
        if (sub->active &&
            sub->layer == event->target_layer &&
            sub->device == event->target_device &&
            (sub->intel_type == event->intel_type || sub->intel_type == DSMIL_INTEL_RAW_DATA)) {
            if (sub->callback) {
                sub->callback(&event_copy);
            }
        }
    }
    
    return 0;
}

int dsmil_intelligence_subscribe(uint8_t layer, uint32_t device,
                                  dsmil_intelligence_type_t intel_type,
                                  dsmil_intelligence_callback_t callback) {
    if (!g_intel_state.initialized) {
        if (dsmil_intelligence_flow_init() != 0) {
            return -1;
        }
    }
    
    if (!callback || layer < 2 || layer > 9) {
        return -1;
    }
    
    if (g_intel_state.num_subscriptions >= MAX_SUBSCRIPTIONS) {
        fprintf(stderr, "ERROR: Maximum subscriptions reached\n");
        return -1;
    }
    
    intelligence_subscription_t *sub = &g_intel_state.subscriptions[g_intel_state.num_subscriptions];
    sub->layer = layer;
    sub->device = device;
    sub->intel_type = intel_type;
    sub->callback = callback;
    sub->active = true;
    
    g_intel_state.num_subscriptions++;
    
    return 0;
}

bool dsmil_intelligence_verify_clearance(uint8_t source_layer, uint8_t target_layer,
                                         uint32_t clearance_mask) {
    // Verify upward flow
    if (target_layer < source_layer) {
        return false;
    }
    
    // Use cross-domain guard for clearance verification
    // (simplified - actual implementation would use dsmil_cross_domain_runtime)
    
    // Basic check: higher layers can receive from lower layers
    return true;
}

int dsmil_intelligence_flow_shutdown(void) {
    if (!g_intel_state.initialized) {
        return 0;
    }
    
    // Clear all subscriptions
    for (uint32_t i = 0; i < g_intel_state.num_subscriptions; i++) {
        g_intel_state.subscriptions[i].active = false;
    }
    
    g_intel_state.num_subscriptions = 0;
    g_intel_state.queue_count = 0;
    g_intel_state.initialized = false;
    
    return 0;
}
