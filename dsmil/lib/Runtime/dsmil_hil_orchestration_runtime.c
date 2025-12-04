/**
 * @file dsmil_hil_orchestration_runtime.c
 * @brief Hardware Integration Layer Orchestration Runtime Implementation
 * 
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_hil_orchestration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_TOPS 13.0f
#define GPU_TOPS 32.0f
#define CPU_TOPS 3.2f
#define MAX_UTILIZATION 0.95f  // 95% max utilization

static struct {
    bool initialized;
    float npu_utilization;
    float gpu_utilization;
    float cpu_utilization;
} g_hil_state = {0};

int dsmil_hil_init(void) {
    if (g_hil_state.initialized) {
        return 0;
    }
    
    g_hil_state.npu_utilization = 0.0f;
    g_hil_state.gpu_utilization = 0.0f;
    g_hil_state.cpu_utilization = 0.0f;
    g_hil_state.initialized = true;
    
    return 0;
}

dsmil_hil_unit_t dsmil_hil_assign_workload(uint32_t device_id, uint8_t layer,
                                            const char *workload_type,
                                            dsmil_hil_unit_t preferred_unit) {
    if (!g_hil_state.initialized) {
        if (dsmil_hil_init() != 0) {
            return DSMIL_HIL_CPU;  // Fallback to CPU
        }
    }
    
    // Device 47 (LLM) prefers GPU for attention
    if (device_id == 47 && workload_type && strstr(workload_type, "llm")) {
        if (g_hil_state.gpu_utilization < MAX_UTILIZATION) {
            return DSMIL_HIL_GPU;
        }
    }
    
    // Device 46 (Quantum) uses CPU (simulation)
    if (device_id == 46) {
        return DSMIL_HIL_CPU;
    }
    
    // Check preferred unit availability
    if (preferred_unit != DSMIL_HIL_NPU && preferred_unit != DSMIL_HIL_GPU && preferred_unit != DSMIL_HIL_CPU) {
        preferred_unit = DSMIL_HIL_CPU;  // Default to CPU
    }
    
    float util = 0.0f;
    dsmil_hil_get_utilization(preferred_unit, &util);
    if (util < MAX_UTILIZATION) {
        return preferred_unit;
    }
    
    // Find least utilized unit
    float npu_util = g_hil_state.npu_utilization;
    float gpu_util = g_hil_state.gpu_utilization;
    float cpu_util = g_hil_state.cpu_utilization;
    
    if (npu_util <= gpu_util && npu_util <= cpu_util && npu_util < MAX_UTILIZATION) {
        return DSMIL_HIL_NPU;
    }
    if (gpu_util <= cpu_util && gpu_util < MAX_UTILIZATION) {
        return DSMIL_HIL_GPU;
    }
    if (cpu_util < MAX_UTILIZATION) {
        return DSMIL_HIL_CPU;
    }
    
    // All units overloaded, return CPU as fallback
    return DSMIL_HIL_CPU;
}

int dsmil_hil_get_utilization(dsmil_hil_unit_t unit, float *utilization) {
    if (!utilization) {
        return -1;
    }
    
    if (!g_hil_state.initialized) {
        if (dsmil_hil_init() != 0) {
            return -1;
        }
    }
    
    switch (unit) {
        case DSMIL_HIL_NPU:
            *utilization = g_hil_state.npu_utilization;
            break;
        case DSMIL_HIL_GPU:
            *utilization = g_hil_state.gpu_utilization;
            break;
        case DSMIL_HIL_CPU:
            *utilization = g_hil_state.cpu_utilization;
            break;
        default:
            return -1;
    }
    
    return 0;
}

bool dsmil_hil_check_availability(dsmil_hil_unit_t unit, float required_tops) {
    if (!g_hil_state.initialized) {
        if (dsmil_hil_init() != 0) {
            return false;
        }
    }
    
    float capacity = 0.0f;
    float utilization = 0.0f;
    
    switch (unit) {
        case DSMIL_HIL_NPU:
            capacity = NPU_TOPS;
            utilization = g_hil_state.npu_utilization;
            break;
        case DSMIL_HIL_GPU:
            capacity = GPU_TOPS;
            utilization = g_hil_state.gpu_utilization;
            break;
        case DSMIL_HIL_CPU:
            capacity = CPU_TOPS;
            utilization = g_hil_state.cpu_utilization;
            break;
        default:
            return false;
    }
    
    float available = capacity * (1.0f - utilization);
    return (available >= required_tops);
}

int dsmil_hil_get_unit_info(dsmil_hil_unit_t unit, dsmil_hil_unit_info_t *info) {
    if (!info) {
        return -1;
    }
    
    if (!g_hil_state.initialized) {
        if (dsmil_hil_init() != 0) {
            return -1;
        }
    }
    
    info->unit = unit;
    
    switch (unit) {
        case DSMIL_HIL_NPU:
            info->tops_capacity = NPU_TOPS;
            info->tops_utilization = g_hil_state.npu_utilization;
            break;
        case DSMIL_HIL_GPU:
            info->tops_capacity = GPU_TOPS;
            info->tops_utilization = g_hil_state.gpu_utilization;
            break;
        case DSMIL_HIL_CPU:
            info->tops_capacity = CPU_TOPS;
            info->tops_utilization = g_hil_state.cpu_utilization;
            break;
        default:
            return -1;
    }
    
    info->available = (info->tops_utilization < MAX_UTILIZATION);
    info->memory_used_bytes = 0;  // Placeholder
    info->memory_total_bytes = 0;  // Placeholder
    
    return 0;
}
