/**
 * @file dsmil_fuzz_telemetry_advanced.h
 * @brief Advanced DSLLVM General-Purpose Fuzzing & Telemetry API
 *
 * Enhanced telemetry API for next-generation fuzzing techniques with
 * rich metadata, ML integration, and high-performance support.
 * General-purpose foundation for any fuzzing target.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_FUZZ_TELEMETRY_ADVANCED_H
#define DSMIL_FUZZ_TELEMETRY_ADVANCED_H

#include "dsmil_fuzz_telemetry.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_ADVANCED_FUZZ Advanced General Fuzzing API
 * @{
 */

/**
 * Fuzzing strategy types
 */
typedef enum {
    DSMIL_FUZZ_STRATEGY_MUTATION = 1,      /**< Standard mutation */
    DSMIL_FUZZ_STRATEGY_CROSSOVER = 2,     /**< Crossover/recombination */
    DSMIL_FUZZ_STRATEGY_GRAMMAR = 3,       /**< Grammar-based generation */
    DSMIL_FUZZ_STRATEGY_DICTIONARY = 4,    /**< Dictionary-based */
    DSMIL_FUZZ_STRATEGY_STRUCTURE = 5,     /**< Structure-aware */
    DSMIL_FUZZ_STRATEGY_ML_GUIDED = 6,     /**< ML-guided mutation */
    DSMIL_FUZZ_STRATEGY_SYMBOLIC = 7,      /**< Symbolic execution */
    DSMIL_FUZZ_STRATEGY_CONCOLIC = 8       /**< Concolic execution */
} dsmil_fuzz_strategy_t;

/**
 * Coverage feedback structure
 */
typedef struct {
    uint64_t input_hash;                   /**< Hash of input that triggered coverage */
    uint32_t new_edges;                    /**< Number of new edges discovered */
    uint32_t new_states;                   /**< Number of new states discovered */
    uint32_t total_edges;                  /**< Total edges covered */
    uint32_t total_states;                 /**< Total states covered */
    uint64_t execution_time_ns;            /**< Execution time in nanoseconds */
    uint64_t memory_peak_bytes;           /**< Peak memory usage */
    double interestingness_score;          /**< ML-computed interestingness (0.0-1.0) */
} dsmil_coverage_feedback_t;

/**
 * Mutation metadata
 */
typedef struct {
    dsmil_fuzz_strategy_t strategy;       /**< Mutation strategy used */
    uint32_t mutation_count;               /**< Number of mutations applied */
    uint32_t seed_input_id;                /**< ID of seed input */
    uint32_t mutation_depth;              /**< Depth in mutation chain */
    const char *mutation_type;             /**< Type of mutation */
    uint64_t mutation_pos;                 /**< Position in input */
    uint64_t mutation_size;                /**< Size of mutation */
    double mutation_score;                 /**< Quality score */
} dsmil_mutation_metadata_t;

/**
 * Advanced telemetry event (extends base event)
 */
typedef struct {
    dsmil_fuzz_telemetry_event_t base;     /**< Base telemetry event */
    
    // Advanced fields
    dsmil_fuzz_strategy_t fuzz_strategy;   /**< Fuzzing strategy used */
    dsmil_mutation_metadata_t mutation;    /**< Mutation metadata */
    
    // Performance metrics
    uint64_t cpu_cycles;                   /**< CPU cycles consumed */
    uint64_t cache_misses;                 /**< Cache misses */
    uint64_t branch_mispredicts;           /**< Branch mispredictions */
    uint64_t tlb_misses;                   /**< TLB misses */
    
    // Coverage metrics
    uint32_t basic_blocks_executed;        /**< Basic blocks executed */
    uint32_t functions_called;              /**< Functions called */
    uint32_t loops_iterated;               /**< Total loop iterations */
    
    // Memory metrics
    uint64_t heap_allocated;               /**< Heap memory allocated */
    uint64_t stack_peak;                   /**< Peak stack usage */
    uint32_t malloc_count;                 /**< Number of malloc calls */
    uint32_t free_count;                   /**< Number of free calls */
    
    // Security metrics
    uint32_t potential_vulns;              /**< Potential vulnerabilities */
    uint32_t sanitizer_findings;           /**< Sanitizer findings */
    uint32_t undefined_behaviors;         /**< Undefined behaviors */
    
    // ML/AI metadata
    double confidence_score;               /**< ML confidence score */
    const char *ml_model_version;          /**< ML model version */
    uint64_t ml_inference_time_ns;         /**< ML inference time */
    
    // Distributed fuzzing
    uint32_t worker_id;                    /**< Worker/thread ID */
    uint32_t generation;                   /**< Generation number */
    uint64_t corpus_size;                  /**< Corpus size */
} dsmil_advanced_fuzz_event_t;

/**
 * Coverage map structure
 */
typedef struct {
    uint32_t *edge_map;                    /**< Edge coverage map */
    uint32_t *state_map;                   /**< State coverage map */
    size_t edge_map_size;                  /**< Size of edge map */
    size_t state_map_size;                 /**< Size of state map */
    uint64_t total_executions;             /**< Total executions */
    uint64_t unique_inputs;                /**< Unique inputs */
} dsmil_coverage_map_t;

/**
 * Initialize advanced telemetry
 *
 * @param config_path Path to YAML config
 * @param ring_buffer_size Ring buffer size
 * @param enable_perf_counters Enable performance counters
 * @param enable_ml Enable ML integration
 * @return 0 on success, negative on error
 */
int dsmil_fuzz_telemetry_advanced_init(const char *config_path,
                                       size_t ring_buffer_size,
                                       int enable_perf_counters,
                                       int enable_ml);

/**
 * Record advanced telemetry event
 */
void dsmil_fuzz_record_advanced_event(const dsmil_advanced_fuzz_event_t *event);

/**
 * Get coverage feedback for input
 */
int dsmil_fuzz_get_coverage_feedback(uint64_t input_hash,
                                     dsmil_coverage_feedback_t *feedback);

/**
 * Update coverage map
 */
int dsmil_fuzz_update_coverage_map(uint64_t input_hash,
                                    const uint32_t *new_edges, size_t new_edges_count,
                                    const uint32_t *new_states, size_t new_states_count);

/**
 * Get coverage map statistics
 */
void dsmil_fuzz_get_coverage_stats(uint32_t *total_edges,
                                   uint32_t *total_states,
                                   uint64_t *unique_inputs);

/**
 * Record mutation metadata
 */
void dsmil_fuzz_record_mutation(const dsmil_mutation_metadata_t *metadata);

/**
 * Get mutation suggestions (ML-guided)
 */
size_t dsmil_fuzz_get_mutation_suggestions(uint32_t seed_input_id,
                                           dsmil_mutation_metadata_t *suggestions,
                                           size_t max_suggestions);

/**
 * Record performance counters
 */
void dsmil_fuzz_record_perf_counters(uint64_t cpu_cycles,
                                     uint64_t cache_misses,
                                     uint64_t branch_mispredicts);

/**
 * Compute input interestingness score (ML)
 */
double dsmil_fuzz_compute_interestingness(uint64_t input_hash,
                                          const dsmil_coverage_feedback_t *coverage_feedback);

/**
 * Export telemetry for ML training
 */
int dsmil_fuzz_export_for_ml(const char *filepath, const char *format);

/**
 * Enable/disable telemetry features
 */
void dsmil_fuzz_set_telemetry_features(uint64_t feature_mask);

/**
 * Get telemetry statistics
 */
void dsmil_fuzz_get_telemetry_stats(uint64_t *total_events,
                                     double *events_per_sec,
                                     double *ring_buffer_utilization);

/**
 * Flush advanced telemetry with compression
 */
int dsmil_fuzz_flush_advanced_events(const char *filepath, int compress);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_FUZZ_TELEMETRY_ADVANCED_H */
