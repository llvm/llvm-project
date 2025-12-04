/**
 * @file dsmil_config_validator.c
 * @brief Configuration Validation Implementation
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil_config_validator.h"
#include "dsmil_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

/* JSON parsing (simplified - in production use proper JSON library) */
static bool is_valid_json(const char *path) {
    if (!path) {
        return false;
    }
    
    FILE *f = fopen(path, "r");
    if (!f) {
        return false;
    }
    
    /* Basic JSON validation: check for opening brace */
    int c = fgetc(f);
    bool valid = (c == '{' || c == '[');
    fclose(f);
    return valid;
}

int dsmil_validate_mission_profiles(const char *profile_path,
                                     dsmil_validation_result_t *result) {
    if (!profile_path || !result) {
        if (result) {
            result->valid = false;
            result->error_message = "Invalid parameters";
            result->error_code = EINVAL;
        }
        return -1;
    }

    result->component = "mission_profiles";
    
    /* Check file exists */
    if (access(profile_path, R_OK) != 0) {
        result->valid = false;
        result->error_message = "Mission profile file not found or not readable";
        result->error_code = errno;
        return -1;
    }

    /* Check JSON validity */
    if (!is_valid_json(profile_path)) {
        result->valid = false;
        result->error_message = "Invalid JSON syntax in mission profile";
        result->error_code = EINVAL;
        return -1;
    }

    /* TODO: Schema validation against mission profile schema */
    /* TODO: Validate profile names, settings, etc. */

    result->valid = true;
    result->error_message = NULL;
    result->error_code = 0;
    return 0;
}

int dsmil_validate_paths(dsmil_validation_result_t *result) {
    if (!result) {
        return -1;
    }

    result->component = "paths";
    result->valid = true;
    result->error_message = NULL;
    result->error_code = 0;

    /* Initialize path system */
    dsmil_paths_init();

    /* Check config directory */
    const char *config_dir = dsmil_get_config_dir();
    if (!dsmil_path_exists(config_dir)) {
        result->valid = false;
        result->error_message = "Configuration directory does not exist";
        result->error_code = ENOENT;
        return -1;
    }

    /* Check bin directory */
    const char *bin_dir = dsmil_get_bin_dir();
    if (!dsmil_path_exists(bin_dir)) {
        result->valid = false;
        result->error_message = "Binary directory does not exist";
        result->error_code = ENOENT;
        return -1;
    }

    /* Check truststore directory */
    const char *truststore_dir = dsmil_get_truststore_dir();
    if (!dsmil_path_exists(truststore_dir)) {
        /* Warning, not error - truststore may be created later */
    }

    /* Check log directory */
    const char *log_dir = dsmil_get_log_dir();
    if (!dsmil_path_exists(log_dir)) {
        /* Warning, not error - log dir may be created at runtime */
    }

    return 0;
}

int dsmil_validate_truststore(const char *truststore_dir,
                               dsmil_validation_result_t *result) {
    if (!result) {
        return -1;
    }

    result->component = "truststore";
    
    const char *dir = truststore_dir ? truststore_dir : dsmil_get_truststore_dir();
    
    if (!dsmil_path_exists(dir)) {
        result->valid = false;
        result->error_message = "Truststore directory does not exist";
        result->error_code = ENOENT;
        return -1;
    }

    /* Check for required certificate files */
    char cert_path[1024];
    const char *certs[] = {"psk_cert.pem", "prk_cert.pem", "rta_cert.pem", NULL};
    
    for (int i = 0; certs[i]; i++) {
        snprintf(cert_path, sizeof(cert_path), "%s/%s", dir, certs[i]);
        if (!dsmil_path_exists(cert_path)) {
            result->valid = false;
            result->error_message = "Missing required certificate file";
            result->error_code = ENOENT;
            return -1;
        }
    }

    /* TODO: Validate certificate chains */
    /* TODO: Check revocation lists */
    /* TODO: Verify signatures */

    result->valid = true;
    result->error_message = NULL;
    result->error_code = 0;
    return 0;
}

int dsmil_validate_classification(dsmil_validation_result_t *result) {
    if (!result) {
        return -1;
    }

    result->component = "classification";
    
    /* TODO: Validate cross-domain gateway configurations */
    /* TODO: Check classification level consistency */
    /* TODO: Verify gateway approval status */

    result->valid = true;
    result->error_message = NULL;
    result->error_code = 0;
    return 0;
}

int dsmil_validate_all(const dsmil_validation_options_t *options,
                       dsmil_validation_result_t *result) {
    if (!result) {
        return -1;
    }

    result->component = "all";
    result->valid = true;
    result->error_message = NULL;
    result->error_code = 0;

    dsmil_validation_result_t component_result = {0};

    /* Validate paths */
    if (!options || options->check_paths) {
        if (dsmil_validate_paths(&component_result) != 0) {
            result->valid = false;
            result->error_message = component_result.error_message;
            return -1;
        }
    }

    /* Validate mission profiles */
    if (!options || options->check_mission_profiles) {
        char config_path[1024];
        const char *profile_file = options && options->config_path ?
            options->config_path : "mission-profiles.json";
        
        if (dsmil_resolve_config(profile_file, config_path, sizeof(config_path))) {
            if (dsmil_validate_mission_profiles(config_path, &component_result) != 0) {
                result->valid = false;
                result->error_message = component_result.error_message;
                return -1;
            }
        }
    }

    /* Validate truststore */
    if (!options || options->check_truststore) {
        if (dsmil_validate_truststore(NULL, &component_result) != 0) {
            /* Truststore validation failure is warning, not error */
            if (options && options->verbose) {
                result->error_message = component_result.error_message;
            }
        }
    }

    /* Validate classification */
    if (!options || options->check_classification) {
        if (dsmil_validate_classification(&component_result) != 0) {
            result->valid = false;
            result->error_message = component_result.error_message;
            return -1;
        }
    }

    return 0;
}

int dsmil_auto_fix_config(const dsmil_validation_options_t *options) {
    if (!options) {
        return -1;
    }

    int fixes = 0;

    /* Create missing directories */
    dsmil_paths_init();
    
    const char *config_dir = dsmil_get_config_dir();
    if (!dsmil_path_exists(config_dir)) {
        if (dsmil_ensure_dir(config_dir, 0755) == 0) {
            fixes++;
        }
    }

    const char *log_dir = dsmil_get_log_dir();
    if (!dsmil_path_exists(log_dir)) {
        if (dsmil_ensure_dir(log_dir, 0755) == 0) {
            fixes++;
        }
    }

    const char *truststore_dir = dsmil_get_truststore_dir();
    if (!dsmil_path_exists(truststore_dir)) {
        if (dsmil_ensure_dir(truststore_dir, 0700) == 0) {
            fixes++;
        }
    }

    return fixes;
}

int dsmil_generate_health_report(const char *output_path,
                                  const dsmil_validation_options_t *options) {
    if (!output_path) {
        return -1;
    }

    FILE *f = fopen(output_path, "w");
    if (!f) {
        return -1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"timestamp\": \"%ld\",\n", time(NULL));
    fprintf(f, "  \"validation_results\": {\n");

    dsmil_validation_result_t result = {0};
    
    /* Validate each component */
    dsmil_validate_paths(&result);
    fprintf(f, "    \"paths\": {\n");
    fprintf(f, "      \"valid\": %s,\n", result.valid ? "true" : "false");
    if (result.error_message) {
        fprintf(f, "      \"error\": \"%s\",\n", result.error_message);
    }
    fprintf(f, "      \"component\": \"%s\"\n", result.component ? result.component : "paths");
    fprintf(f, "    },\n");

    dsmil_validate_truststore(NULL, &result);
    fprintf(f, "    \"truststore\": {\n");
    fprintf(f, "      \"valid\": %s,\n", result.valid ? "true" : "false");
    if (result.error_message) {
        fprintf(f, "      \"error\": \"%s\",\n", result.error_message);
    }
    fprintf(f, "      \"component\": \"%s\"\n", result.component ? result.component : "truststore");
    fprintf(f, "    }\n");

    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}
