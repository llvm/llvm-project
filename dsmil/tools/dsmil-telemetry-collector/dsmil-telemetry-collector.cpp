/**
 * @file dsmil-telemetry-collector.cpp
 * @brief DSLLVM Telemetry Collector Tool
 *
 * Collects and exports runtime telemetry data.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_telemetry_export.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <unistd.h>

static bool running = true;

static void signal_handler(int sig) {
    (void)sig;
    running = false;
}

static void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --format=FORMAT         Export format (prometheus, otel, json)\n");
    fprintf(stderr, "  --endpoint=URL          Endpoint URL (for OpenTelemetry)\n");
    fprintf(stderr, "  --port=PORT             Port number (for Prometheus)\n");
    fprintf(stderr, "  --output=FILE            Output file (for JSON)\n");
    fprintf(stderr, "  --enable-performance    Enable performance metrics\n");
    fprintf(stderr, "  --enable-security       Enable security events\n");
    fprintf(stderr, "  --enable-operational    Enable operational metrics\n");
    fprintf(stderr, "  --help, -h              Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --format=prometheus --port=9090\n", prog_name);
    fprintf(stderr, "  %s --format=otel --endpoint=http://otel:4317\n", prog_name);
    fprintf(stderr, "  %s --format=json --output=/var/log/dsmil/telemetry.json\n", prog_name);
}

int main(int argc, char **argv) {
    dsmil_telemetry_options_t options = {0};
    options.format = DSMIL_TELEMETRY_JSON;
    options.enable_performance = true;
    options.enable_security = true;
    options.enable_operational = true;

    static struct option long_options[] = {
        {"format", required_argument, 0, 'f'},
        {"endpoint", required_argument, 0, 'e'},
        {"port", required_argument, 0, 'p'},
        {"output", required_argument, 0, 'o'},
        {"enable-performance", no_argument, 0, 'P'},
        {"enable-security", no_argument, 0, 'S'},
        {"enable-operational", no_argument, 0, 'O'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "f:e:p:o:PSOh", long_options, NULL)) != -1) {
        switch (c) {
        case 'f':
            if (strcmp(optarg, "prometheus") == 0) {
                options.format = DSMIL_TELEMETRY_PROMETHEUS;
            } else if (strcmp(optarg, "otel") == 0 || strcmp(optarg, "opentelemetry") == 0) {
                options.format = DSMIL_TELEMETRY_OPENTELEMETRY;
            } else if (strcmp(optarg, "json") == 0) {
                options.format = DSMIL_TELEMETRY_JSON;
            }
            break;
        case 'e':
            options.endpoint = optarg;
            break;
        case 'p':
            options.port = atoi(optarg);
            break;
        case 'o':
            options.output_file = optarg;
            break;
        case 'P':
            options.enable_performance = true;
            break;
        case 'S':
            options.enable_security = true;
            break;
        case 'O':
            options.enable_operational = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    dsmil_paths_init();

    if (dsmil_telemetry_init(&options) != 0) {
        fprintf(stderr, "Failed to initialize telemetry export\n");
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("Telemetry collector started (format: %d)\n", options.format);
    printf("Press Ctrl+C to stop\n");

    /* Main collection loop */
    while (running) {
        /* TODO: Collect metrics from shared memory or socket */
        /* For now, just wait */
        sleep(1);
    }

    dsmil_telemetry_shutdown();
    printf("\nTelemetry collector stopped\n");

    return 0;
}
