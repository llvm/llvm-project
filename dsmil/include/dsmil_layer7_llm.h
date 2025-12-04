/**
 * @file dsmil_layer7_llm.h
 * @brief Layer 7 (EXTENDED) - Device 47 Advanced AI/ML Runtime Support
 * 
 * Provides runtime support for primary LLM workloads on Device 47:
 * - Memory management (40 GB max Layer 7 budget)
 * - KV cache optimization
 * - INT8 quantization enforcement
 * - Model lifecycle management
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_LAYER7_LLM_H
#define DSMIL_LAYER7_LLM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "dsmil_int8_quantization.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_LAYER7_LLM Layer 7 Device 47 LLM Runtime
 * @{
 */

/**
 * @brief Device 47 LLM context
 */
typedef struct {
    uint32_t device_id;           // 47
    uint8_t layer;                // 7
    uint64_t memory_budget_bytes; // From Layer 7 pool (max 40 GB)
    uint64_t memory_used_bytes;
    bool int8_quantized;          // Must be INT8 per MLOps pipeline
    uint32_t model_id;
    const char *model_name;
    uint64_t kv_cache_size_bytes;
    uint32_t context_length;
    float quantization_accuracy;  // Must be >95% per MLOps requirements
} dsmil_device47_llm_ctx_t;

/**
 * @brief Initialize Device 47 LLM runtime
 * 
 * @param memory_budget Maximum memory budget from Layer 7 pool (default: 40 GB)
 * @return 0 on success, negative on error
 */
int dsmil_device47_llm_init(uint64_t memory_budget);

/**
 * @brief Load INT8-quantized LLM model
 * 
 * @param model_path Path to INT8 model file
 * @param ctx Output context
 * @return 0 on success, negative on error
 */
int dsmil_device47_llm_load(const char *model_path, dsmil_device47_llm_ctx_t *ctx);

/**
 * @brief Verify INT8 quantization (must be >95% accuracy retention)
 * 
 * @param ctx LLM context
 * @return true if quantization is valid, false otherwise
 */
bool dsmil_device47_verify_int8_quantization(const dsmil_device47_llm_ctx_t *ctx);

/**
 * @brief Get INT8 quantization parameters for model
 * 
 * @param ctx LLM context
 * @param params Output quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_device47_get_int8_params(const dsmil_device47_llm_ctx_t *ctx,
                                    dsmil_int8_params_t *params);

/**
 * @brief Perform INT8 matrix multiplication for attention/FFN layers
 * 
 * @param ctx LLM context
 * @param A Input matrix A (INT8)
 * @param B Weight matrix B (INT8)
 * @param output Output matrix (FP32)
 * @param layer_type Layer type ("attention", "ffn", "embedding")
 * @return 0 on success, negative on error
 */
int dsmil_device47_int8_matmul(const dsmil_device47_llm_ctx_t *ctx,
                                const int8_t *A, const int8_t *B,
                                float *output, const char *layer_type);

/**
 * @brief Get current memory usage
 * 
 * @param ctx LLM context
 * @return Memory used in bytes
 */
uint64_t dsmil_device47_get_memory_usage(const dsmil_device47_llm_ctx_t *ctx);

/**
 * @brief Check if memory budget is exceeded
 * 
 * @param ctx LLM context
 * @return true if within budget, false if exceeded
 */
bool dsmil_device47_check_memory_budget(const dsmil_device47_llm_ctx_t *ctx);

/**
 * @brief Set KV cache size
 * 
 * @param ctx LLM context
 * @param kv_cache_size KV cache size in bytes
 * @return 0 on success, negative on error
 */
int dsmil_device47_set_kv_cache_size(dsmil_device47_llm_ctx_t *ctx, uint64_t kv_cache_size);

/**
 * @brief Unload model and free resources
 * 
 * @param ctx LLM context
 * @return 0 on success, negative on error
 */
int dsmil_device47_llm_unload(dsmil_device47_llm_ctx_t *ctx);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_LAYER7_LLM_H */
