/**
 * @file dsmil_memory_budget_runtime.c
 * @brief Dynamic Memory Budget Runtime Implementation
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_memory_budget.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define TOTAL_AVAILABLE (62ULL * 1024 * 1024 * 1024)  // 62 GB
#define LAYER2_MAX (4ULL * 1024 * 1024 * 1024)       // 4 GB
#define LAYER3_MAX (6ULL * 1024 * 1024 * 1024)       // 6 GB
#define LAYER4_MAX (8ULL * 1024 * 1024 * 1024)       // 8 GB
#define LAYER5_MAX (10ULL * 1024 * 1024 * 1024)      // 10 GB
#define LAYER6_MAX (12ULL * 1024 * 1024 * 1024)      // 12 GB
#define LAYER7_MAX (40ULL * 1024 * 1024 * 1024)      // 40 GB
#define LAYER8_MAX (8ULL * 1024 * 1024 * 1024)       // 8 GB
#define LAYER9_MAX (12ULL * 1024 * 1024 * 1024)      // 12 GB

static struct {
    bool initialized;
    dsmil_memory_usage_t usage;
    dsmil_memory_budgets_t budgets;
    pthread_mutex_t mutex;
} g_memory_state = {0};

static uint64_t get_layer_max(uint8_t layer) {
    switch (layer) {
        case 2: return LAYER2_MAX;
        case 3: return LAYER3_MAX;
        case 4: return LAYER4_MAX;
        case 5: return LAYER5_MAX;
        case 6: return LAYER6_MAX;
        case 7: return LAYER7_MAX;
        case 8: return LAYER8_MAX;
        case 9: return LAYER9_MAX;
        default: return 0;
    }
}

int dsmil_memory_budget_init(void) {
    if (g_memory_state.initialized) {
        return 0;
    }
    
    memset(&g_memory_state, 0, sizeof(g_memory_state));
    
    // Initialize budgets
    g_memory_state.budgets.layer2_max_bytes = LAYER2_MAX;
    g_memory_state.budgets.layer3_max_bytes = LAYER3_MAX;
    g_memory_state.budgets.layer4_max_bytes = LAYER4_MAX;
    g_memory_state.budgets.layer5_max_bytes = LAYER5_MAX;
    g_memory_state.budgets.layer6_max_bytes = LAYER6_MAX;
    g_memory_state.budgets.layer7_max_bytes = LAYER7_MAX;
    g_memory_state.budgets.layer8_max_bytes = LAYER8_MAX;
    g_memory_state.budgets.layer9_max_bytes = LAYER9_MAX;
    g_memory_state.budgets.total_available = TOTAL_AVAILABLE;
    
    // Initialize mutex
    if (pthread_mutex_init(&g_memory_state.mutex, NULL) != 0) {
        return -1;
    }
    
    g_memory_state.initialized = true;
    return 0;
}

int dsmil_memory_budget_get_budgets(dsmil_memory_budgets_t *budgets) {
    if (!budgets) {
        return -1;
    }
    
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return -1;
        }
    }
    
    *budgets = g_memory_state.budgets;
    return 0;
}

void *dsmil_memory_allocate(uint8_t layer, uint64_t size_bytes) {
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return NULL;
        }
    }
    
    if (layer < 2 || layer > 9 || size_bytes == 0) {
        return NULL;
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    
    // Check layer budget
    uint64_t layer_max = get_layer_max(layer);
    uint64_t layer_used = 0;
    switch (layer) {
        case 2: layer_used = g_memory_state.usage.layer2_used_bytes; break;
        case 3: layer_used = g_memory_state.usage.layer3_used_bytes; break;
        case 4: layer_used = g_memory_state.usage.layer4_used_bytes; break;
        case 5: layer_used = g_memory_state.usage.layer5_used_bytes; break;
        case 6: layer_used = g_memory_state.usage.layer6_used_bytes; break;
        case 7: layer_used = g_memory_state.usage.layer7_used_bytes; break;
        case 8: layer_used = g_memory_state.usage.layer8_used_bytes; break;
        case 9: layer_used = g_memory_state.usage.layer9_used_bytes; break;
    }
    
    if (layer_used + size_bytes > layer_max) {
        pthread_mutex_unlock(&g_memory_state.mutex);
        return NULL;  // Would exceed layer budget
    }
    
    // Check global constraint
    if (g_memory_state.usage.total_used_bytes + size_bytes > TOTAL_AVAILABLE) {
        pthread_mutex_unlock(&g_memory_state.mutex);
        return NULL;  // Would exceed global constraint
    }
    
    // Allocate memory
    void *ptr = malloc(size_bytes);
    if (!ptr) {
        pthread_mutex_unlock(&g_memory_state.mutex);
        return NULL;
    }
    
    // Update usage
    switch (layer) {
        case 2: g_memory_state.usage.layer2_used_bytes += size_bytes; break;
        case 3: g_memory_state.usage.layer3_used_bytes += size_bytes; break;
        case 4: g_memory_state.usage.layer4_used_bytes += size_bytes; break;
        case 5: g_memory_state.usage.layer5_used_bytes += size_bytes; break;
        case 6: g_memory_state.usage.layer6_used_bytes += size_bytes; break;
        case 7: g_memory_state.usage.layer7_used_bytes += size_bytes; break;
        case 8: g_memory_state.usage.layer8_used_bytes += size_bytes; break;
        case 9: g_memory_state.usage.layer9_used_bytes += size_bytes; break;
    }
    g_memory_state.usage.total_used_bytes += size_bytes;
    
    pthread_mutex_unlock(&g_memory_state.mutex);
    return ptr;
}

void dsmil_memory_free(uint8_t layer, void *ptr, uint64_t size_bytes) {
    if (!ptr || size_bytes == 0 || layer < 2 || layer > 9) {
        return;
    }
    
    if (!g_memory_state.initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    
    free(ptr);
    
    // Update usage
    switch (layer) {
        case 2:
            if (g_memory_state.usage.layer2_used_bytes >= size_bytes) {
                g_memory_state.usage.layer2_used_bytes -= size_bytes;
            }
            break;
        case 3:
            if (g_memory_state.usage.layer3_used_bytes >= size_bytes) {
                g_memory_state.usage.layer3_used_bytes -= size_bytes;
            }
            break;
        case 4:
            if (g_memory_state.usage.layer4_used_bytes >= size_bytes) {
                g_memory_state.usage.layer4_used_bytes -= size_bytes;
            }
            break;
        case 5:
            if (g_memory_state.usage.layer5_used_bytes >= size_bytes) {
                g_memory_state.usage.layer5_used_bytes -= size_bytes;
            }
            break;
        case 6:
            if (g_memory_state.usage.layer6_used_bytes >= size_bytes) {
                g_memory_state.usage.layer6_used_bytes -= size_bytes;
            }
            break;
        case 7:
            if (g_memory_state.usage.layer7_used_bytes >= size_bytes) {
                g_memory_state.usage.layer7_used_bytes -= size_bytes;
            }
            break;
        case 8:
            if (g_memory_state.usage.layer8_used_bytes >= size_bytes) {
                g_memory_state.usage.layer8_used_bytes -= size_bytes;
            }
            break;
        case 9:
            if (g_memory_state.usage.layer9_used_bytes >= size_bytes) {
                g_memory_state.usage.layer9_used_bytes -= size_bytes;
            }
            break;
    }
    
    if (g_memory_state.usage.total_used_bytes >= size_bytes) {
        g_memory_state.usage.total_used_bytes -= size_bytes;
    }
    
    pthread_mutex_unlock(&g_memory_state.mutex);
}

bool dsmil_memory_check_budget(uint8_t layer, uint64_t size_bytes) {
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return false;
        }
    }
    
    if (layer < 2 || layer > 9) {
        return false;
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    
    uint64_t layer_max = get_layer_max(layer);
    uint64_t layer_used = 0;
    switch (layer) {
        case 2: layer_used = g_memory_state.usage.layer2_used_bytes; break;
        case 3: layer_used = g_memory_state.usage.layer3_used_bytes; break;
        case 4: layer_used = g_memory_state.usage.layer4_used_bytes; break;
        case 5: layer_used = g_memory_state.usage.layer5_used_bytes; break;
        case 6: layer_used = g_memory_state.usage.layer6_used_bytes; break;
        case 7: layer_used = g_memory_state.usage.layer7_used_bytes; break;
        case 8: layer_used = g_memory_state.usage.layer8_used_bytes; break;
        case 9: layer_used = g_memory_state.usage.layer9_used_bytes; break;
    }
    
    bool within_budget = (layer_used + size_bytes <= layer_max) &&
                         (g_memory_state.usage.total_used_bytes + size_bytes <= TOTAL_AVAILABLE);
    
    pthread_mutex_unlock(&g_memory_state.mutex);
    return within_budget;
}

int dsmil_memory_get_usage(dsmil_memory_usage_t *usage) {
    if (!usage) {
        return -1;
    }
    
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return -1;
        }
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    *usage = g_memory_state.usage;
    pthread_mutex_unlock(&g_memory_state.mutex);
    
    return 0;
}

bool dsmil_memory_verify_global_constraint(void) {
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return false;
        }
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    bool within_constraint = g_memory_state.usage.total_used_bytes <= TOTAL_AVAILABLE;
    pthread_mutex_unlock(&g_memory_state.mutex);
    
    return within_constraint;
}

int dsmil_memory_get_layer_available(uint8_t layer, uint64_t *available) {
    if (!available || layer < 2 || layer > 9) {
        return -1;
    }
    
    if (!g_memory_state.initialized) {
        if (dsmil_memory_budget_init() != 0) {
            return -1;
        }
    }
    
    pthread_mutex_lock(&g_memory_state.mutex);
    
    uint64_t layer_max = get_layer_max(layer);
    uint64_t layer_used = 0;
    switch (layer) {
        case 2: layer_used = g_memory_state.usage.layer2_used_bytes; break;
        case 3: layer_used = g_memory_state.usage.layer3_used_bytes; break;
        case 4: layer_used = g_memory_state.usage.layer4_used_bytes; break;
        case 5: layer_used = g_memory_state.usage.layer5_used_bytes; break;
        case 6: layer_used = g_memory_state.usage.layer6_used_bytes; break;
        case 7: layer_used = g_memory_state.usage.layer7_used_bytes; break;
        case 8: layer_used = g_memory_state.usage.layer8_used_bytes; break;
        case 9: layer_used = g_memory_state.usage.layer9_used_bytes; break;
    }
    
    *available = (layer_max > layer_used) ? (layer_max - layer_used) : 0;
    
    pthread_mutex_unlock(&g_memory_state.mutex);
    return 0;
}
