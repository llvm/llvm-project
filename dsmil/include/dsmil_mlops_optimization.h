/**
 * @file dsmil_mlops_optimization.h
 * @brief MLOps Pipeline Optimization Support
 * 
 * Provides compile-time and runtime support for MLOps pipeline:
 * - INT8 quantization enforcement (mandatory)
 * - Pruning (50% sparsity target)
 * - Knowledge distillation (7B → 1.5B)
 * - Flash Attention 2 for transformers
 * - Model fusion and checkpointing
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_MLOPS_OPTIMIZATION_H
#define DSMIL_MLOPS_OPTIMIZATION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_MLOPS MLOps Pipeline Optimization
 * @{
 */

/**
 * @brief MLOps optimization targets (from comprehensive plan)
 */
typedef struct {
    float quantization_speedup;      // 4.0× (FP32 → INT8)
    float pruning_speedup;           // 2.5× (50% sparsity)
    float distillation_speedup;      // 4.0× (7B → 1.5B)
    float flash_attention_speedup;   // 2.0× (transformers)
    float combined_minimum;           // 12.0× minimum
    float combined_target;            // 30.0× target (bridge gap)
    float combined_maximum;           // 60.0× maximum
} dsmil_mlops_targets_t;

/**
 * @brief Model optimization status
 */
typedef struct {
    bool int8_quantized;
    float quantization_accuracy_retention;  // Must be >95%
    bool pruned;
    float pruning_sparsity;                 // Target: 50%
    bool distilled;
    bool flash_attention_enabled;
    float combined_speedup;
    bool meets_requirements;
} dsmil_mlops_status_t;

/**
 * @brief Get default MLOps optimization targets
 * 
 * @param targets Output targets
 * @return 0 on success, negative on error
 */
int dsmil_mlops_get_default_targets(dsmil_mlops_targets_t *targets);

/**
 * @brief Verify model meets MLOps requirements
 * 
 * @param model_path Path to model
 * @param targets Optimization targets
 * @param status Output optimization status
 * @return true if model meets requirements, false otherwise
 */
bool dsmil_mlops_verify_model(const char *model_path,
                              const dsmil_mlops_targets_t *targets,
                              dsmil_mlops_status_t *status);

/**
 * @brief Check INT8 quantization (must be >95% accuracy retention)
 * 
 * @param model_path Path to INT8 model
 * @param accuracy_retention Output accuracy retention percentage
 * @return true if quantization is valid, false otherwise
 */
bool dsmil_mlops_verify_int8_quantization(const char *model_path, float *accuracy_retention);

/**
 * @brief Verify pruning sparsity (target: 50%)
 * 
 * @param model_path Path to pruned model
 * @param sparsity Output sparsity percentage
 * @return true if sparsity meets target, false otherwise
 */
bool dsmil_mlops_verify_pruning(const char *model_path, float *sparsity);

/**
 * @brief Calculate combined optimization speedup
 * 
 * @param status Optimization status
 * @param speedup Output combined speedup multiplier
 * @return 0 on success, negative on error
 */
int dsmil_mlops_calculate_speedup(const dsmil_mlops_status_t *status, float *speedup);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_MLOPS_OPTIMIZATION_H */
