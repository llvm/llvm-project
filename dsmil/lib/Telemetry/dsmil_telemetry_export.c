/**
 * @file dsmil_telemetry_export.c
 * @brief Runtime Telemetry Export Implementation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_telemetry_export.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define MAX_METRICS 1000
#define MAX_LABELS 16

static struct {
    bool initialized;
    dsmil_telemetry_format_t format;
    FILE *output_file;
    const char *endpoint;
    int port;
    bool enable_performance;
    bool enable_security;
    bool enable_operational;
} telemetry_state = {0};

static void write_prometheus_metric(const char *name, const char *type,
                                     double value, const char **labels) {
    if (!telemetry_state.output_file) {
        return;
    }

    fprintf(telemetry_state.output_file, "# TYPE %s %s\n", name, type);
    fprintf(telemetry_state.output_file, "%s", name);

    if (labels) {
        fprintf(telemetry_state.output_file, "{");
        for (int i = 0; labels[i] && labels[i + 1]; i += 2) {
            if (i > 0) {
                fprintf(telemetry_state.output_file, ",");
            }
            fprintf(telemetry_state.output_file, "%s=\"%s\"", labels[i], labels[i + 1]);
        }
        fprintf(telemetry_state.output_file, "}");
    }

    fprintf(telemetry_state.output_file, " %.6f\n", value);
}

static void write_json_metric(const char *name, const char *type,
                               double value, const char **labels) {
    if (!telemetry_state.output_file) {
        return;
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);

    fprintf(telemetry_state.output_file, "{\n");
    fprintf(telemetry_state.output_file, "  \"timestamp\": %ld.%06ld,\n", tv.tv_sec, tv.tv_usec);
    fprintf(telemetry_state.output_file, "  \"metric\": \"%s\",\n", name);
    fprintf(telemetry_state.output_file, "  \"type\": \"%s\",\n", type);
    fprintf(telemetry_state.output_file, "  \"value\": %.6f", value);

    if (labels) {
        fprintf(telemetry_state.output_file, ",\n  \"labels\": {\n");
        for (int i = 0; labels[i] && labels[i + 1]; i += 2) {
            if (i > 0) {
                fprintf(telemetry_state.output_file, ",\n");
            }
            fprintf(telemetry_state.output_file, "    \"%s\": \"%s\"", labels[i], labels[i + 1]);
        }
        fprintf(telemetry_state.output_file, "\n  }");
    }

    fprintf(telemetry_state.output_file, "\n}\n");
}

int dsmil_telemetry_init(const dsmil_telemetry_options_t *options) {
    if (!options || telemetry_state.initialized) {
        return -1;
    }

    telemetry_state.format = options->format;
    telemetry_state.endpoint = options->endpoint;
    telemetry_state.port = options->port;
    telemetry_state.enable_performance = options->enable_performance;
    telemetry_state.enable_security = options->enable_security;
    telemetry_state.enable_operational = options->enable_operational;

    if (options->output_file) {
        telemetry_state.output_file = fopen(options->output_file, "a");
        if (!telemetry_state.output_file) {
            fprintf(stderr, "ERROR: Failed to open telemetry output file: %s\n",
                    options->output_file);
            return -1;
        }
    } else {
        const char *log_dir = dsmil_get_log_dir();
        if (log_dir) {
            char log_path[1024];
            snprintf(log_path, sizeof(log_path), "%s/telemetry.log", log_dir);
            telemetry_state.output_file = fopen(log_path, "a");
            if (!telemetry_state.output_file) {
                fprintf(stderr, "WARNING: Failed to open telemetry log: %s, using stderr\n",
                        log_path);
                telemetry_state.output_file = stderr; /* Fallback */
            }
        } else {
            telemetry_state.output_file = stderr; /* Fallback */
        }
    }

    telemetry_state.initialized = true;
    return 0;
}

void dsmil_telemetry_record_counter(const char *name, uint64_t value, const char **labels) {
    if (!telemetry_state.initialized || !name) {
        return;
    }

    if (!telemetry_state.enable_performance) {
        return;
    }

    switch (telemetry_state.format) {
    case DSMIL_TELEMETRY_PROMETHEUS:
        write_prometheus_metric(name, "counter", (double)value, labels);
        break;
    case DSMIL_TELEMETRY_JSON:
        write_json_metric(name, "counter", (double)value, labels);
        break;
    default:
        break;
    }
}

void dsmil_telemetry_record_gauge(const char *name, double value, const char **labels) {
    if (!telemetry_state.initialized || !name) {
        return;
    }

    if (!telemetry_state.enable_performance) {
        return;
    }

    switch (telemetry_state.format) {
    case DSMIL_TELEMETRY_PROMETHEUS:
        write_prometheus_metric(name, "gauge", value, labels);
        break;
    case DSMIL_TELEMETRY_JSON:
        write_json_metric(name, "gauge", value, labels);
        break;
    default:
        break;
    }
}

void dsmil_telemetry_record_histogram(const char *name, double value, const char **labels) {
    if (!telemetry_state.initialized || !name) {
        return;
    }

    if (!telemetry_state.enable_performance) {
        return;
    }

    switch (telemetry_state.format) {
    case DSMIL_TELEMETRY_PROMETHEUS:
        write_prometheus_metric(name, "histogram", value, labels);
        break;
    case DSMIL_TELEMETRY_JSON:
        write_json_metric(name, "histogram", value, labels);
        break;
    default:
        break;
    }
}

void dsmil_telemetry_record_security_event(const char *event_type,
                                            int severity,
                                            const char *details) {
    if (!telemetry_state.initialized || !event_type) {
        return;
    }

    if (!telemetry_state.enable_security) {
        return;
    }

    if (telemetry_state.format == DSMIL_TELEMETRY_JSON && telemetry_state.output_file) {
        struct timeval tv;
        gettimeofday(&tv, NULL);

        fprintf(telemetry_state.output_file, "{\n");
        fprintf(telemetry_state.output_file, "  \"timestamp\": %ld.%06ld,\n", tv.tv_sec, tv.tv_usec);
        fprintf(telemetry_state.output_file, "  \"event_type\": \"security\",\n");
        fprintf(telemetry_state.output_file, "  \"event\": \"%s\",\n", event_type);
        fprintf(telemetry_state.output_file, "  \"severity\": %d", severity);
        if (details) {
            fprintf(telemetry_state.output_file, ",\n  \"details\": %s", details);
        }
        fprintf(telemetry_state.output_file, "\n}\n");
    }
}

void dsmil_telemetry_record_operational(const char *name, uint64_t value, const char **labels) {
    if (!telemetry_state.initialized || !name) {
        return;
    }

    if (!telemetry_state.enable_operational) {
        return;
    }

    switch (telemetry_state.format) {
    case DSMIL_TELEMETRY_PROMETHEUS:
        write_prometheus_metric(name, "gauge", (double)value, labels);
        break;
    case DSMIL_TELEMETRY_JSON:
        write_json_metric(name, "operational", (double)value, labels);
        break;
    default:
        break;
    }
}

int dsmil_telemetry_flush(void) {
    if (!telemetry_state.initialized || !telemetry_state.output_file) {
        return -1;
    }

    fflush(telemetry_state.output_file);
    return 0;
}

int dsmil_telemetry_shutdown(void) {
    if (!telemetry_state.initialized) {
        return -1;
    }

    dsmil_telemetry_flush();

    if (telemetry_state.output_file && telemetry_state.output_file != stderr) {
        fclose(telemetry_state.output_file);
    }

    telemetry_state.initialized = false;
    return 0;
}
