/**
 * @file dsmil_int8_quantization.h
 * @brief Advanced INT8 Quantization Runtime for DSMIL
 * 
 * Provides comprehensive INT8 quantization support for:
 * - LLM model quantization (mandatory INT8 per MLOps pipeline)
 * - Matrix operations (INT8 GEMM, MatMul)
 * - Calibration and accuracy validation (>95% retention required)
 * - Hardware acceleration (NPU/GPU INT8 TOPS)
 * - Dynamic quantization for inference
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_INT8_QUANTIZATION_H
#define DSMIL_INT8_QUANTIZATION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_INT8 Advanced INT8 Quantization
 * @{
 */

/**
 * @brief Quantization scheme types
 */
typedef enum {
    DSMIL_INT8_SYMMETRIC,      // Symmetric quantization (zero point = 0)
    DSMIL_INT8_ASYMMETRIC,     // Asymmetric quantization (zero point != 0)
    DSMIL_INT8_PER_TENSOR,     // Single scale/zero-point per tensor
    DSMIL_INT8_PER_CHANNEL,    // Per-channel quantization (conv/linear layers)
    DSMIL_INT8_DYNAMIC          // Dynamic quantization (runtime scales)
} dsmil_int8_scheme_t;

/**
 * @brief INT8 quantization parameters
 */
typedef struct {
    float scale;                // Quantization scale
    int8_t zero_point;          // Zero point (for asymmetric)
    int8_t qmin;               // Minimum quantized value (-128)
    int8_t qmax;               // Maximum quantized value (127)
    dsmil_int8_scheme_t scheme;
    bool per_channel;          // Per-channel quantization
    uint32_t num_channels;     // Number of channels (if per_channel)
    float *channel_scales;      // Per-channel scales (if per_channel)
    int8_t *channel_zero_points; // Per-channel zero points
} dsmil_int8_params_t;

/**
 * @brief INT8 calibration statistics
 */
typedef struct {
    float min_value;           // Observed minimum FP32 value
    float max_value;           // Observed maximum FP32 value
    float mean_value;          // Mean FP32 value
    float std_dev;            // Standard deviation
    uint64_t sample_count;     // Number of calibration samples
    float kl_divergence;       // KL divergence (for calibration quality)
} dsmil_int8_calibration_stats_t;

/**
 * @brief INT8 accuracy metrics
 */
typedef struct {
    float fp32_accuracy;       // FP32 baseline accuracy
    float int8_accuracy;       // INT8 quantized accuracy
    float accuracy_retention;  // int8_accuracy / fp32_accuracy (must be >0.95)
    float mse_error;          // Mean squared error
    float max_error;          // Maximum per-element error
    bool meets_requirement;    // accuracy_retention >= 0.95
} dsmil_int8_accuracy_metrics_t;

/**
 * @brief INT8 matrix operation context
 */
typedef struct {
    uint32_t M;                // Rows
    uint32_t N;                // Columns
    uint32_t K;                // Inner dimension
    dsmil_int8_params_t *A_params;  // Matrix A quantization params
    dsmil_int8_params_t *B_params;  // Matrix B quantization params
    dsmil_int8_params_t *C_params;  // Output matrix C quantization params
    bool use_hardware_accel;   // Use NPU/GPU INT8 acceleration
} dsmil_int8_matmul_ctx_t;

/**
 * @brief Calibrate quantization parameters from FP32 data
 * 
 * @param fp32_data FP32 input data
 * @param data_size Number of elements
 * @param scheme Quantization scheme
 * @param per_channel Enable per-channel quantization
 * @param num_channels Number of channels (if per_channel)
 * @param params Output quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_int8_calibrate(const float *fp32_data, size_t data_size,
                         dsmil_int8_scheme_t scheme, bool per_channel,
                         uint32_t num_channels,
                         dsmil_int8_params_t *params);

/**
 * @brief Quantize FP32 tensor to INT8
 * 
 * @param fp32_data Input FP32 data
 * @param int8_data Output INT8 data
 * @param data_size Number of elements
 * @param params Quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_int8_quantize(const float *fp32_data, int8_t *int8_data,
                        size_t data_size, const dsmil_int8_params_t *params);

/**
 * @brief Dequantize INT8 tensor to FP32
 * 
 * @param int8_data Input INT8 data
 * @param fp32_data Output FP32 data
 * @param data_size Number of elements
 * @param params Quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_int8_dequantize(const int8_t *int8_data, float *fp32_data,
                          size_t data_size, const dsmil_int8_params_t *params);

/**
 * @brief INT8 matrix multiplication (GEMM)
 * 
 * C = A * B (all INT8, output may be INT32 or INT8)
 * 
 * @param ctx Matrix multiplication context
 * @param A INT8 matrix A (M x K)
 * @param B INT8 matrix B (K x N)
 * @param C Output matrix C (M x N, INT32 or INT8)
 * @param use_int32_output Use INT32 accumulator (true) or INT8 output (false)
 * @return 0 on success, negative on error
 */
int dsmil_int8_gemm(const dsmil_int8_matmul_ctx_t *ctx,
                    const int8_t *A, const int8_t *B, void *C,
                    bool use_int32_output);

/**
 * @brief INT8 matrix multiplication with bias and activation
 * 
 * C = activation((A * B) + bias)
 * 
 * @param ctx Matrix multiplication context
 * @param A INT8 matrix A
 * @param B INT8 matrix B
 * @param bias FP32 bias vector (N elements)
 * @param C Output matrix C (FP32)
 * @param activation_type Activation function ("relu", "gelu", "none")
 * @return 0 on success, negative on error
 */
int dsmil_int8_matmul_with_bias(const dsmil_int8_matmul_ctx_t *ctx,
                                 const int8_t *A, const int8_t *B,
                                 const float *bias, float *C,
                                 const char *activation_type);

/**
 * @brief Validate INT8 quantization accuracy
 * 
 * Compares INT8 quantized model output with FP32 baseline.
 * Must achieve >95% accuracy retention per MLOps requirements.
 * 
 * @param fp32_model_path Path to FP32 baseline model
 * @param int8_model_path Path to INT8 quantized model
 * @param test_dataset_path Path to test dataset
 * @param metrics Output accuracy metrics
 * @return 0 if validation passes, negative on error
 */
int dsmil_int8_validate_accuracy(const char *fp32_model_path,
                                const char *int8_model_path,
                                const char *test_dataset_path,
                                dsmil_int8_accuracy_metrics_t *metrics);

/**
 * @brief Get calibration statistics
 * 
 * @param fp32_data FP32 calibration data
 * @param data_size Number of elements
 * @param stats Output calibration statistics
 * @return 0 on success, negative on error
 */
int dsmil_int8_get_calibration_stats(const float *fp32_data, size_t data_size,
                                     dsmil_int8_calibration_stats_t *stats);

/**
 * @brief Optimize quantization parameters for accuracy
 * 
 * Uses calibration statistics to find optimal scale/zero-point
 * that maximizes accuracy retention.
 * 
 * @param stats Calibration statistics
 * @param target_retention Target accuracy retention (default: 0.95)
 * @param params Output optimized quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_int8_optimize_params(const dsmil_int8_calibration_stats_t *stats,
                                float target_retention,
                                dsmil_int8_params_t *params);

/**
 * @brief Convert FP32 model weights to INT8
 * 
 * Performs per-layer or per-channel quantization of model weights.
 * 
 * @param fp32_weights FP32 weight tensor
 * @param weight_shape Weight tensor shape [dims...]
 * @param num_dims Number of dimensions
 * @param layer_type Layer type ("linear", "conv2d", "embedding")
 * @param int8_weights Output INT8 weights
 * @param params Output quantization parameters
 * @return 0 on success, negative on error
 */
int dsmil_int8_quantize_weights(const float *fp32_weights,
                                const uint32_t *weight_shape, uint32_t num_dims,
                                const char *layer_type,
                                int8_t *int8_weights,
                                dsmil_int8_params_t *params);

/**
 * @brief Dynamic quantization for inference
 * 
 * Quantizes activations at runtime with dynamic scales.
 * 
 * @param fp32_activations FP32 activation tensor
 * @param activation_size Number of elements
 * @param int8_activations Output INT8 activations
 * @param scale Output dynamic scale
 * @param zero_point Output dynamic zero point
 * @return 0 on success, negative on error
 */
int dsmil_int8_dynamic_quantize(const float *fp32_activations,
                                size_t activation_size,
                                int8_t *int8_activations,
                                float *scale, int8_t *zero_point);

/**
 * @brief Free quantization parameters
 * 
 * @param params Parameters to free
 */
void dsmil_int8_free_params(dsmil_int8_params_t *params);

/**
 * @brief Get INT8 hardware acceleration capabilities
 * 
 * @param npu_tops Output NPU INT8 TOPS capacity
 * @param gpu_tops Output GPU INT8 TOPS capacity
 * @param cpu_tops Output CPU INT8 TOPS capacity
 * @return 0 on success, negative on error
 */
int dsmil_int8_get_hardware_caps(float *npu_tops, float *gpu_tops, float *cpu_tops);

/**
 * @brief Estimate INT8 inference speedup vs FP32
 * 
 * @param model_size Model size in parameters
 * @param batch_size Batch size
 * @param sequence_length Sequence length (for transformers)
 * @param speedup Output estimated speedup multiplier
 * @return 0 on success, negative on error
 */
int dsmil_int8_estimate_speedup(uint64_t model_size, uint32_t batch_size,
                                uint32_t sequence_length, float *speedup);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_INT8_QUANTIZATION_H */
