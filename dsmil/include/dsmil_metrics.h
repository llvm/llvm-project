/**
 * @file dsmil_metrics.h
 * @brief Compile-Time Performance Metrics API
 *
 * Provides metrics collection for compilation performance, pass execution,
 * and optimization effectiveness.
 *
 * Version: 1.7.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_METRICS_H
#define DSMIL_METRICS_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_METRICS Compile-Time Metrics
 * @{
 */

/**
 * @brief Pass execution metrics
 */
typedef struct {
    const char *pass_name;
    uint64_t execution_time_ns;
    uint64_t memory_peak_bytes;
    uint64_t memory_avg_bytes;
    uint64_t ir_size_before;
    uint64_t ir_size_after;
    bool success;
} dsmil_pass_metrics_t;

/**
 * @brief Feature impact metrics
 */
typedef struct {
    const char *feature_name;
    uint64_t overhead_ns;
    uint64_t memory_overhead_bytes;
    double speedup_estimate;
} dsmil_feature_metrics_t;

/**
 * @brief Build metrics summary
 */
typedef struct {
    uint64_t total_compile_time_ns;
    uint64_t total_memory_peak_bytes;
    uint32_t num_passes;
    uint32_t num_functions;
    uint64_t code_size_bytes;
    double optimization_effectiveness;
    dsmil_pass_metrics_t *passes;
    dsmil_feature_metrics_t *features;
} dsmil_build_metrics_t;

/**
 * @brief Initialize metrics collection
 *
 * @param output_path Path to output JSON file (NULL for stdout)
 * @return 0 on success, -1 on error
 */
int dsmil_metrics_init(const char *output_path);

/**
 * @brief Start timing a pass
 *
 * @param pass_name Name of the pass
 * @return Pass ID (>= 0) on success, -1 on error
 */
int dsmil_metrics_start_pass(const char *pass_name);

/**
 * @brief End timing a pass
 *
 * @param pass_id Pass ID from dsmil_metrics_start_pass
 * @param ir_size_before IR size before pass
 * @param ir_size_after IR size after pass
 * @return 0 on success, -1 on error
 */
int dsmil_metrics_end_pass(int pass_id, uint64_t ir_size_before, uint64_t ir_size_after);

/**
 * @brief Record feature impact
 *
 * @param feature_name Name of the feature
 * @param overhead_ns Overhead in nanoseconds
 * @param memory_overhead_bytes Memory overhead in bytes
 */
void dsmil_metrics_record_feature(const char *feature_name,
                                   uint64_t overhead_ns,
                                   uint64_t memory_overhead_bytes);

/**
 * @brief Finalize metrics collection and write output
 *
 * @return 0 on success, -1 on error
 */
int dsmil_metrics_finalize(void);

/**
 * @brief Get current build metrics
 *
 * @param metrics Output metrics structure
 * @return 0 on success, -1 on error
 */
int dsmil_metrics_get_build_metrics(dsmil_build_metrics_t *metrics);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_METRICS_H */
