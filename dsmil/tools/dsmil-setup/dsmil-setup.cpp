/**
 * @file dsmil-setup.cpp
 * @brief DSLLVM Setup Wizard Tool
 *
 * Interactive wizard for DSLLVM installation and configuration.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_setup.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

static void print_usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --non-interactive        Non-interactive mode\n");
    fprintf(stderr, "  --profile=PROFILE        Mission profile template\n");
    fprintf(stderr, "  --output=FILE            Output configuration file\n");
    fprintf(stderr, "  --verify                 Verify existing installation\n");
    fprintf(stderr, "  --fix                    Fix common issues\n");
    fprintf(stderr, "  --prefix=PREFIX          Installation prefix\n");
    fprintf(stderr, "  --help, -h               Show this help message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s                      # Interactive wizard\n", prog_name);
    fprintf(stderr, "  %s --non-interactive --profile=cyber_defence\n", prog_name);
    fprintf(stderr, "  %s --verify\n", prog_name);
    fprintf(stderr, "  %s --fix\n", prog_name);
}

int main(int argc, char **argv) {
    dsmil_setup_options_t options = {0};
    options.interactive = true;

    static struct option long_options[] = {
        {"non-interactive", no_argument, 0, 'n'},
        {"profile", required_argument, 0, 'p'},
        {"output", required_argument, 0, 'o'},
        {"verify", no_argument, 0, 'v'},
        {"fix", no_argument, 0, 'f'},
        {"prefix", required_argument, 0, 'P'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "np:o:vfP:h", long_options, NULL)) != -1) {
        switch (c) {
        case 'n':
            options.interactive = false;
            break;
        case 'p':
            options.profile_template = optarg;
            break;
        case 'o':
            options.output_path = optarg;
            break;
        case 'v':
            options.verify_only = true;
            break;
        case 'f':
            options.fix_issues = true;
            break;
        case 'P':
            options.prefix = optarg;
            setenv("DSMIL_PREFIX", optarg, 1);
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

    if (options.verify_only) {
        return dsmil_setup_verify(&options) == 0 ? 0 : 1;
    }

    if (options.fix_issues && !options.interactive) {
        int fixes = dsmil_setup_fix(&options);
        return fixes >= 0 ? 0 : 1;
    }

    return dsmil_setup_wizard(&options) == 0 ? 0 : 1;
}
