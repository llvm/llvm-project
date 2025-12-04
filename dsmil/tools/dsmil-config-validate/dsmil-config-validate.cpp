/**
 * @file dsmil-config-validate.cpp
 * @brief DSLLVM Configuration Validation Tool
 *
 * Validates mission profiles, paths, truststore, and other configuration.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_config_validator.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

static void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --all                    Validate all configuration components\n");
    fprintf(stderr, "  --mission-profiles       Validate mission profile configuration\n");
    fprintf(stderr, "  --truststore             Validate truststore configuration\n");
    fprintf(stderr, "  --paths                  Validate path configuration\n");
    fprintf(stderr, "  --classification        Validate classification configuration\n");
    fprintf(stderr, "  --auto-fix               Automatically fix common issues\n");
    fprintf(stderr, "  --report=FILE            Generate health report JSON\n");
    fprintf(stderr, "  --verbose, -v            Verbose output\n");
    fprintf(stderr, "  --help, -h               Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --all\n", prog_name);
    fprintf(stderr, "  %s --mission-profiles --truststore\n", prog_name);
    fprintf(stderr, "  %s --auto-fix --report=health.json\n", prog_name);
}

int main(int argc, char **argv) {
    dsmil_validation_options_t options = {0};
    options.check_paths = true;
    options.check_truststore = true;
    options.check_mission_profiles = true;
    options.check_classification = true;

    const char *report_path = NULL;
    bool validate_all = false;
    bool auto_fix = false;

    static struct option long_options[] = {
        {"all", no_argument, 0, 'a'},
        {"mission-profiles", no_argument, 0, 'm'},
        {"truststore", no_argument, 0, 't'},
        {"paths", no_argument, 0, 'p'},
        {"classification", no_argument, 0, 'c'},
        {"auto-fix", no_argument, 0, 'f'},
        {"report", required_argument, 0, 'r'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "amtpcfr:vh", long_options, NULL)) != -1) {
        switch (c) {
        case 'a':
            validate_all = true;
            break;
        case 'm':
            options.check_mission_profiles = true;
            options.check_paths = false;
            options.check_truststore = false;
            options.check_classification = false;
            break;
        case 't':
            options.check_truststore = true;
            options.check_paths = false;
            options.check_mission_profiles = false;
            options.check_classification = false;
            break;
        case 'p':
            options.check_paths = true;
            options.check_truststore = false;
            options.check_mission_profiles = false;
            options.check_classification = false;
            break;
        case 'c':
            options.check_classification = true;
            options.check_paths = false;
            options.check_truststore = false;
            options.check_mission_profiles = false;
            break;
        case 'f':
            auto_fix = true;
            options.auto_fix = true;
            break;
        case 'r':
            report_path = optarg;
            break;
        case 'v':
            options.verbose = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    if (validate_all) {
        options.check_paths = true;
        options.check_truststore = true;
        options.check_mission_profiles = true;
        options.check_classification = true;
    }

    /* Initialize path system */
    dsmil_paths_init();

    /* Auto-fix if requested */
    if (auto_fix) {
        int fixes = dsmil_auto_fix_config(&options);
        if (fixes > 0) {
            printf("Fixed %d configuration issue(s)\n", fixes);
        }
    }

    /* Validate configuration */
    dsmil_validation_result_t result = {0};
    int ret = dsmil_validate_all(&options, &result);

    if (options.verbose || !result.valid) {
        if (result.valid) {
            printf("✓ Configuration validation passed\n");
            if (result.component) {
                printf("  Component: %s\n", result.component);
            }
        } else {
            printf("✗ Configuration validation failed\n");
            if (result.component) {
                printf("  Component: %s\n", result.component);
            }
            if (result.error_message) {
                printf("  Error: %s\n", result.error_message);
            }
            if (result.error_code != 0) {
                printf("  Error code: %d\n", result.error_code);
            }
        }
    }

    /* Generate report if requested */
    if (report_path) {
        if (dsmil_generate_health_report(report_path, &options) == 0) {
            printf("Health report written to: %s\n", report_path);
        } else {
            fprintf(stderr, "Failed to generate health report\n");
            ret = 1;
        }
    }

    return result.valid ? 0 : 1;
}
