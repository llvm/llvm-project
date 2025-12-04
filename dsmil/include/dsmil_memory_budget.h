/**
 * @file dsmil_memory_budget.h
 * @brief Dynamic Memory Budget Management
 * 
 * Manages 62 GB memory pool across 9 operational layers:
 * - Layer 2: 4 GB max
 * - Layer 3: 6 GB max
 * - Layer 4: 8 GB max
 * - Layer 5: 10 GB max
 * - Layer 6: 12 GB max
 * - Layer 7: 40 GB max (PRIMARY AI LAYER)
 * - Layer 8: 8 GB max
 * - Layer 9: 12 GB max
 * 
 * Budgets are maximums, not hard reservations.
 * Runtime: sum(active_layer_usage) ≤ 62 GB
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_MEMORY_BUDGET_H
#define DSMIL_MEMORY_BUDGET_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_MEMORY Memory Budget Management
 * @{
 */

/**
 * @brief Layer memory budgets (maximums)
 */
typedef struct {
    uint64_t layer2_max_bytes;  // 4 GB
    uint64_t layer3_max_bytes;  // 6 GB
    uint64_t layer4_max_bytes;  // 8 GB
    uint64_t layer5_max_bytes;  // 10 GB
    uint64_t layer6_max_bytes;  // 12 GB
    uint64_t layer7_max_bytes;  // 40 GB (PRIMARY AI)
    uint64_t layer8_max_bytes;  // 8 GB
    uint64_t layer9_max_bytes;  // 12 GB
    uint64_t total_available;   // 62 GB
} dsmil_memory_budgets_t;

/**
 * @brief Current memory usage per layer
 */
typedef struct {
    uint64_t layer2_used_bytes;
    uint64_t layer3_used_bytes;
    uint64_t layer4_used_bytes;
    uint64_t layer5_used_bytes;
    uint64_t layer6_used_bytes;
    uint64_t layer7_used_bytes;
    uint64_t layer8_used_bytes;
    uint64_t layer9_used_bytes;
    uint64_t total_used_bytes;
} dsmil_memory_usage_t;

/**
 * @brief Initialize memory budget system
 * 
 * @return 0 on success, negative on error
 */
int dsmil_memory_budget_init(void);

/**
 * @brief Get layer memory budgets
 * 
 * @param budgets Output budgets
 * @return 0 on success, negative on error
 */
int dsmil_memory_budget_get_budgets(dsmil_memory_budgets_t *budgets);

/**
 * @brief Allocate memory from layer budget
 * 
 * @param layer Layer number (2-9)
 * @param size_bytes Requested size
 * @return Pointer to allocated memory, NULL on failure
 */
void *dsmil_memory_allocate(uint8_t layer, uint64_t size_bytes);

/**
 * @brief Free memory and update layer usage
 * 
 * @param layer Layer number (2-9)
 * @param ptr Pointer to memory
 * @param size_bytes Size of memory
 */
void dsmil_memory_free(uint8_t layer, void *ptr, uint64_t size_bytes);

/**
 * @brief Check if allocation would exceed budget
 * 
 * @param layer Layer number (2-9)
 * @param size_bytes Requested size
 * @return true if within budget, false if would exceed
 */
bool dsmil_memory_check_budget(uint8_t layer, uint64_t size_bytes);

/**
 * @brief Get current memory usage statistics
 * 
 * @param usage Output usage statistics
 * @return 0 on success, negative on error
 */
int dsmil_memory_get_usage(dsmil_memory_usage_t *usage);

/**
 * @brief Verify global memory constraint (sum ≤ 62 GB)
 * 
 * @return true if within constraint, false if exceeded
 */
bool dsmil_memory_verify_global_constraint(void);

/**
 * @brief Get available memory for layer
 * 
 * @param layer Layer number (2-9)
 * @param available Output available bytes
 * @return 0 on success, negative on error
 */
int dsmil_memory_get_layer_available(uint8_t layer, uint64_t *available);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_MEMORY_BUDGET_H */
