/**
 * @file dsmil_hil_orchestration.h
 * @brief Hardware Integration Layer (HIL) Orchestration
 * 
 * Orchestrates workloads across Intel Core Ultra 7 165H:
 * - NPU: 13.0 TOPS INT8 (continuous inference)
 * - GPU: 32.0 TOPS INT8 (dense math, vision, LLM attention)
 * - CPU: 3.2 TOPS INT8 (control plane, scalar workloads)
 * 
 * Total: 48.2 TOPS INT8 physical
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_HIL_ORCHESTRATION_H
#define DSMIL_HIL_ORCHESTRATION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_HIL Hardware Integration Layer
 * @{
 */

/**
 * @brief Hardware compute unit types
 */
typedef enum {
    DSMIL_HIL_NPU,   // Neural Processing Unit (13.0 TOPS)
    DSMIL_HIL_GPU,   // Arc Graphics (32.0 TOPS)
    DSMIL_HIL_CPU    // CPU P/E cores + AMX (3.2 TOPS)
} dsmil_hil_unit_t;

/**
 * @brief Hardware unit capabilities
 */
typedef struct {
    dsmil_hil_unit_t unit;
    float tops_capacity;      // TOPS capacity
    float tops_utilization;  // Current utilization (0.0-1.0)
    bool available;           // Available for new workloads
    uint64_t memory_used_bytes;
    uint64_t memory_total_bytes;
} dsmil_hil_unit_info_t;

/**
 * @brief Workload assignment to hardware unit
 * 
 * @param device_id DSMIL device ID (0-103, 255)
 * @param layer Layer number (2-9)
 * @param workload_type Workload type (inference, training, etc.)
 * @param preferred_unit Preferred hardware unit
 * @return Assigned hardware unit
 */
dsmil_hil_unit_t dsmil_hil_assign_workload(uint32_t device_id, uint8_t layer,
                                            const char *workload_type,
                                            dsmil_hil_unit_t preferred_unit);

/**
 * @brief Get current TOPS utilization per hardware unit
 * 
 * @param unit Hardware unit
 * @param utilization Output utilization (0.0-1.0)
 * @return 0 on success, negative on error
 */
int dsmil_hil_get_utilization(dsmil_hil_unit_t unit, float *utilization);

/**
 * @brief Check if hardware unit can accept new workload
 * 
 * @param unit Hardware unit
 * @param required_tops Required TOPS
 * @return true if available, false if overloaded
 */
bool dsmil_hil_check_availability(dsmil_hil_unit_t unit, float required_tops);

/**
 * @brief Get hardware unit information
 * 
 * @param unit Hardware unit
 * @param info Output unit information
 * @return 0 on success, negative on error
 */
int dsmil_hil_get_unit_info(dsmil_hil_unit_t unit, dsmil_hil_unit_info_t *info);

/**
 * @brief Initialize HIL orchestration system
 * 
 * @return 0 on success, negative on error
 */
int dsmil_hil_init(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_HIL_ORCHESTRATION_H */
