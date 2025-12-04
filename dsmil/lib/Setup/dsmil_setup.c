/**
 * @file dsmil_setup.c
 * @brief Setup Wizard Implementation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_setup.h"
#include "dsmil_paths.h"
#include "dsmil_config_validator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static void print_step(const char *step) {
    printf("\n=== %s ===\n", step);
}

static int detect_dsmil_clang(char *path, size_t path_size) {
    const char *locations[] = {
        "/usr/bin/dsmil-clang",
        "/usr/local/bin/dsmil-clang",
        "/opt/dsmil/bin/dsmil-clang",
        NULL
    };

    for (int i = 0; locations[i]; i++) {
        if (access(locations[i], F_OK) == 0) {
            strncpy(path, locations[i], path_size - 1);
            path[path_size - 1] = '\0';
            return 0;
        }
    }

    /* Check PATH */
    const char *path_env = getenv("PATH");
    if (path_env) {
        char *path_copy = strdup(path_env);
        if (path_copy) {
            char *token = strtok(path_copy, ":");
            while (token) {
                char test_path[1024];
                snprintf(test_path, sizeof(test_path), "%s/dsmil-clang", token);
                if (access(test_path, F_OK) == 0) {
                    strncpy(path, test_path, path_size - 1);
                    path[path_size - 1] = '\0';
                    free(path_copy);
                    return 0;
                }
                token = strtok(NULL, ":");
            }
            free(path_copy);
        }
    }

    return -1;
}

int dsmil_setup_detect_installation(char *prefix, size_t prefix_size) {
    char clang_path[1024];
    if (detect_dsmil_clang(clang_path, sizeof(clang_path)) != 0) {
        return -1;
    }

    /* Extract prefix from path */
    if (strstr(clang_path, "/opt/dsmil/bin")) {
        strncpy(prefix, "/opt/dsmil", prefix_size - 1);
    } else if (strstr(clang_path, "/usr/local/bin")) {
        strncpy(prefix, "/usr/local", prefix_size - 1);
    } else if (strstr(clang_path, "/usr/bin")) {
        strncpy(prefix, "/usr", prefix_size - 1);
    } else {
        /* Try to extract from path */
        char *bin_pos = strstr(clang_path, "/bin/dsmil-clang");
        if (bin_pos) {
            size_t len = bin_pos - clang_path;
            if (len < prefix_size) {
                strncpy(prefix, clang_path, len);
                prefix[len] = '\0';
            }
        }
    }

    prefix[prefix_size - 1] = '\0';
    return 0;
}

static int generate_mission_profile_template(const char *template_name, const char *output_path) {
    FILE *f = fopen(output_path, "w");
    if (!f) {
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": \"1.0\",\n");
    fprintf(f, "  \"profiles\": {\n");

    if (strcmp(template_name, "border_ops") == 0) {
        fprintf(f, "    \"border_ops\": {\n");
        fprintf(f, "      \"description\": \"Border operations profile\",\n");
        fprintf(f, "      \"pipeline\": \"hardened\",\n");
        fprintf(f, "      \"telemetry_level\": \"minimal\",\n");
        fprintf(f, "      \"constant_time_enforcement\": \"strict\"\n");
        fprintf(f, "    }\n");
    } else if (strcmp(template_name, "cyber_defence") == 0) {
        fprintf(f, "    \"cyber_defence\": {\n");
        fprintf(f, "      \"description\": \"Cyber defence profile\",\n");
        fprintf(f, "      \"pipeline\": \"enhanced\",\n");
        fprintf(f, "      \"telemetry_level\": \"full\",\n");
        fprintf(f, "      \"constant_time_enforcement\": \"standard\"\n");
        fprintf(f, "    }\n");
    } else {
        fprintf(f, "    \"default\": {\n");
        fprintf(f, "      \"description\": \"Default profile\",\n");
        fprintf(f, "      \"pipeline\": \"standard\",\n");
        fprintf(f, "      \"telemetry_level\": \"standard\",\n");
        fprintf(f, "      \"constant_time_enforcement\": \"standard\"\n");
        fprintf(f, "    }\n");
    }

    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

int dsmil_setup_generate_config(const char *template_name, const char *output_path) {
    if (!template_name || !output_path) {
        return -1;
    }

    return generate_mission_profile_template(template_name, output_path);
}

int dsmil_setup_verify(const dsmil_setup_options_t *options) {
    dsmil_validation_options_t val_options = {0};
    val_options.check_paths = true;
    val_options.check_truststore = true;
    val_options.check_mission_profiles = true;
    val_options.check_classification = true;
    val_options.verbose = true;

    dsmil_validation_result_t result = {0};
    int ret = dsmil_validate_all(&val_options, &result);

    if (result.valid) {
        printf("✓ Installation verification passed\n");
        return 0;
    } else {
        printf("✗ Installation verification failed\n");
        if (result.error_message) {
            printf("  Error: %s\n", result.error_message);
        }
        return -1;
    }
}

int dsmil_setup_fix(const dsmil_setup_options_t *options) {
    dsmil_validation_options_t val_options = {0};
    val_options.auto_fix = true;

    int fixes = dsmil_auto_fix_config(&val_options);
    printf("Fixed %d issue(s)\n", fixes);
    return fixes;
}

static int interactive_wizard_step_1_detect(void) {
    print_step("Step 1: Detecting DSLLVM Installation");
    
    char prefix[1024];
    if (dsmil_setup_detect_installation(prefix, sizeof(prefix)) == 0) {
        printf("Found DSLLVM installation at: %s\n", prefix);
        return 0;
    } else {
        printf("DSLLVM installation not found in standard locations.\n");
        printf("Please specify installation prefix: ");
        if (fgets(prefix, sizeof(prefix), stdin)) {
            /* Remove newline */
            size_t len = strlen(prefix);
            if (len > 0 && prefix[len - 1] == '\n') {
                prefix[len - 1] = '\0';
            }
            setenv("DSMIL_PREFIX", prefix, 1);
            return 0;
        }
        return -1;
    }
}

static int interactive_wizard_step_2_profile(void) {
    print_step("Step 2: Mission Profile Selection");
    
    printf("Select mission profile:\n");
    printf("  1. border_ops (Border operations - maximum security)\n");
    printf("  2. cyber_defence (Cyber defence - AI-enhanced)\n");
    printf("  3. exercise_only (Training exercises - relaxed)\n");
    printf("  4. lab_research (Laboratory research - experimental)\n");
    printf("Enter choice (1-4): ");

    char choice[10];
    if (!fgets(choice, sizeof(choice), stdin)) {
        return -1;
    }

    const char *profile = NULL;
    switch (choice[0]) {
    case '1': profile = "border_ops"; break;
    case '2': profile = "cyber_defence"; break;
    case '3': profile = "exercise_only"; break;
    case '4': profile = "lab_research"; break;
    default:
        printf("Invalid choice, using default\n");
        profile = "cyber_defence";
    }

    char config_path[1024];
    const char *config_dir = dsmil_get_config_dir();
    snprintf(config_path, sizeof(config_path), "%s/mission-profiles.json", config_dir);

    printf("Generating mission profile: %s\n", profile);
    return dsmil_setup_generate_config(profile, config_path);
}

static int interactive_wizard_step_3_paths(void) {
    print_step("Step 3: Path Configuration");
    
    dsmil_paths_init();
    
    printf("Current path configuration:\n");
    printf("  Prefix: %s\n", dsmil_get_prefix());
    printf("  Config: %s\n", dsmil_get_config_dir());
    printf("  Binaries: %s\n", dsmil_get_bin_dir());
    printf("  Truststore: %s\n", dsmil_get_truststore_dir());
    
    printf("\nCreate missing directories? (y/n): ");
    char response[10];
    if (fgets(response, sizeof(response), stdin) && response[0] == 'y') {
        dsmil_validation_options_t options = {0};
        options.auto_fix = true;
        dsmil_auto_fix_config(&options);
    }

    return 0;
}

static int interactive_wizard_step_4_verify(void) {
    print_step("Step 4: Verification");
    
    return dsmil_setup_verify(NULL);
}

int dsmil_setup_wizard(const dsmil_setup_options_t *options) {
    if (!options || options->interactive) {
        printf("DSLLVM Setup Wizard\n");
        printf("===================\n\n");

        if (interactive_wizard_step_1_detect() != 0) {
            return -1;
        }

        if (interactive_wizard_step_2_profile() != 0) {
            return -1;
        }

        if (interactive_wizard_step_3_paths() != 0) {
            return -1;
        }

        if (interactive_wizard_step_4_verify() != 0) {
            printf("\nWarning: Some verification checks failed\n");
        }

        printf("\n✓ Setup complete!\n");
        return 0;
    } else {
        /* Non-interactive mode */
        if (options->profile_template) {
            char config_path[1024];
            snprintf(config_path, sizeof(config_path), "%s/mission-profiles.json",
                     dsmil_get_config_dir());
            if (dsmil_setup_generate_config(options->profile_template, config_path) != 0) {
                return -1;
            }
        }

        if (options->fix_issues) {
            dsmil_setup_fix(options);
        }

        if (options->verify_only) {
            return dsmil_setup_verify(options);
        }

        return 0;
    }
}
