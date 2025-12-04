/**
 * @file dsmil_metrics.c
 * @brief Compile-Time Metrics Implementation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include "dsmil_metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

#define MAX_PASSES 256
#define MAX_FEATURES 64

static struct {
    bool initialized;
    FILE *output_file;
    const char *output_path;
    dsmil_pass_metrics_t passes[MAX_PASSES];
    dsmil_feature_metrics_t features[MAX_FEATURES];
    uint32_t num_passes;
    uint32_t num_features;
    struct timespec start_time;
    uint64_t total_memory_peak;
} metrics_state = {0};

static uint64_t get_memory_usage(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (uint64_t)usage.ru_maxrss * 1024; /* Convert KB to bytes */
    }
    return 0;
}

static uint64_t timespec_to_ns(const struct timespec *ts) {
    return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

int dsmil_metrics_init(const char *output_path) {
    if (metrics_state.initialized) {
        return 0; /* Already initialized */
    }

    metrics_state.output_path = output_path;
    
    if (output_path) {
        metrics_state.output_file = fopen(output_path, "w");
        if (!metrics_state.output_file) {
            fprintf(stderr, "ERROR: Failed to open metrics output file: %s\n", output_path);
            return -1;
        }
    } else {
        metrics_state.output_file = stdout;
    }

    clock_gettime(CLOCK_MONOTONIC, &metrics_state.start_time);
    metrics_state.initialized = true;
    metrics_state.num_passes = 0;
    metrics_state.num_features = 0;
    metrics_state.total_memory_peak = 0;

    fprintf(metrics_state.output_file, "{\n");
    fprintf(metrics_state.output_file, "  \"build_metrics\": {\n");
    fprintf(metrics_state.output_file, "    \"passes\": [\n");

    return 0;
}

int dsmil_metrics_start_pass(const char *pass_name) {
    if (!metrics_state.initialized || !pass_name) {
        return -1;
    }

    if (metrics_state.num_passes >= MAX_PASSES) {
        return -1;
    }

    int pass_id = metrics_state.num_passes++;
    dsmil_pass_metrics_t *pass = &metrics_state.passes[pass_id];
    
    pass->pass_name = strdup(pass_name);
    if (!pass->pass_name) {
        metrics_state.num_passes--; /* Rollback on allocation failure */
        return -1;
    }
    pass->execution_time_ns = 0;
    pass->memory_peak_bytes = 0;
    pass->memory_avg_bytes = 0;
    pass->ir_size_before = 0;
    pass->ir_size_after = 0;
    pass->success = false;

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pass->execution_time_ns = timespec_to_ns(&start);

    uint64_t mem = get_memory_usage();
    if (mem > metrics_state.total_memory_peak) {
        metrics_state.total_memory_peak = mem;
    }
    pass->memory_peak_bytes = mem;

    return pass_id;
}

int dsmil_metrics_end_pass(int pass_id, uint64_t ir_size_before, uint64_t ir_size_after) {
    if (!metrics_state.initialized || pass_id < 0 || pass_id >= metrics_state.num_passes) {
        return -1;
    }

    dsmil_pass_metrics_t *pass = &metrics_state.passes[pass_id];
    
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    uint64_t end_ns = timespec_to_ns(&end);
    pass->execution_time_ns = end_ns - pass->execution_time_ns;

    uint64_t mem = get_memory_usage();
    if (mem > pass->memory_peak_bytes) {
        pass->memory_peak_bytes = mem;
    }
    pass->memory_avg_bytes = (pass->memory_peak_bytes + mem) / 2;
    pass->ir_size_before = ir_size_before;
    pass->ir_size_after = ir_size_after;
    pass->success = true;

    /* Write pass metrics to JSON */
    if (pass_id > 0) {
        fprintf(metrics_state.output_file, ",\n");
    }
    fprintf(metrics_state.output_file, "      {\n");
    fprintf(metrics_state.output_file, "        \"name\": \"%s\",\n", pass->pass_name);
    fprintf(metrics_state.output_file, "        \"execution_time_ns\": %lu,\n", pass->execution_time_ns);
    fprintf(metrics_state.output_file, "        \"memory_peak_bytes\": %lu,\n", pass->memory_peak_bytes);
    fprintf(metrics_state.output_file, "        \"memory_avg_bytes\": %lu,\n", pass->memory_avg_bytes);
    fprintf(metrics_state.output_file, "        \"ir_size_before\": %lu,\n", pass->ir_size_before);
    fprintf(metrics_state.output_file, "        \"ir_size_after\": %lu,\n", pass->ir_size_after);
    fprintf(metrics_state.output_file, "        \"success\": %s\n", pass->success ? "true" : "false");
    fprintf(metrics_state.output_file, "      }");

    return 0;
}

void dsmil_metrics_record_feature(const char *feature_name,
                                   uint64_t overhead_ns,
                                   uint64_t memory_overhead_bytes) {
    if (!metrics_state.initialized || !feature_name) {
        return;
    }

    if (metrics_state.num_features >= MAX_FEATURES) {
        return;
    }

    dsmil_feature_metrics_t *feature = &metrics_state.features[metrics_state.num_features++];
    feature->feature_name = strdup(feature_name);
    feature->overhead_ns = overhead_ns;
    feature->memory_overhead_bytes = memory_overhead_bytes;
    feature->speedup_estimate = 0.0;
}

int dsmil_metrics_finalize(void) {
    if (!metrics_state.initialized) {
        return -1;
    }

    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    uint64_t total_time = timespec_to_ns(&end_time) - timespec_to_ns(&metrics_state.start_time);

    fprintf(metrics_state.output_file, "\n    ],\n");
    fprintf(metrics_state.output_file, "    \"features\": [\n");

    for (uint32_t i = 0; i < metrics_state.num_features; i++) {
        if (i > 0) {
            fprintf(metrics_state.output_file, ",\n");
        }
        dsmil_feature_metrics_t *feature = &metrics_state.features[i];
        fprintf(metrics_state.output_file, "      {\n");
        fprintf(metrics_state.output_file, "        \"name\": \"%s\",\n", feature->feature_name);
        fprintf(metrics_state.output_file, "        \"overhead_ns\": %lu,\n", feature->overhead_ns);
        fprintf(metrics_state.output_file, "        \"memory_overhead_bytes\": %lu,\n", feature->memory_overhead_bytes);
        fprintf(metrics_state.output_file, "        \"speedup_estimate\": %.2f\n", feature->speedup_estimate);
        fprintf(metrics_state.output_file, "      }");
    }

    fprintf(metrics_state.output_file, "\n    ],\n");
    fprintf(metrics_state.output_file, "    \"summary\": {\n");
    fprintf(metrics_state.output_file, "      \"total_compile_time_ns\": %lu,\n", total_time);
    fprintf(metrics_state.output_file, "      \"total_memory_peak_bytes\": %lu,\n", metrics_state.total_memory_peak);
    fprintf(metrics_state.output_file, "      \"num_passes\": %u,\n", metrics_state.num_passes);
    fprintf(metrics_state.output_file, "      \"num_features\": %u\n", metrics_state.num_features);
    fprintf(metrics_state.output_file, "    }\n");
    fprintf(metrics_state.output_file, "  }\n");
    fprintf(metrics_state.output_file, "}\n");

    /* Free allocated pass names */
    for (uint32_t i = 0; i < metrics_state.num_passes; i++) {
        if (metrics_state.passes[i].pass_name) {
            free((void *)metrics_state.passes[i].pass_name);
            metrics_state.passes[i].pass_name = NULL;
        }
    }

    if (metrics_state.output_file && metrics_state.output_file != stdout) {
        fclose(metrics_state.output_file);
    }

    metrics_state.initialized = false;
    metrics_state.num_passes = 0;
    metrics_state.num_features = 0;
    return 0;
}

int dsmil_metrics_get_build_metrics(dsmil_build_metrics_t *metrics) {
    if (!metrics_state.initialized || !metrics) {
        return -1;
    }

    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    uint64_t total_time = timespec_to_ns(&end_time) - timespec_to_ns(&metrics_state.start_time);

    metrics->total_compile_time_ns = total_time;
    metrics->total_memory_peak_bytes = metrics_state.total_memory_peak;
    metrics->num_passes = metrics_state.num_passes;
    metrics->num_functions = 0; /* TODO: Track function count */
    metrics->code_size_bytes = 0; /* TODO: Track code size */
    metrics->optimization_effectiveness = 0.0; /* TODO: Calculate */
    metrics->passes = metrics_state.passes;
    metrics->features = metrics_state.features;

    return 0;
}
