/**
 * @file dsmil_config_validator.h
 * @brief Configuration Validation API for DSLLVM
 *
 * Provides validation functions for mission profiles, paths, truststore,
 * and other DSLLVM configuration components.
 *
 * Version: 1.7.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_CONFIG_VALIDATOR_H
#define DSMIL_CONFIG_VALIDATOR_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_CONFIG_VALIDATION Configuration Validation
 * @{
 */

/**
 * @brief Validation result structure
 */
typedef struct {
    bool valid;
    const char *error_message;
    const char *component;
    int error_code;
} dsmil_validation_result_t;

/**
 * @brief Validation options
 */
typedef struct {
    bool auto_fix;
    bool verbose;
    bool check_paths;
    bool check_truststore;
    bool check_mission_profiles;
    bool check_classification;
    const char *config_path;
} dsmil_validation_options_t;

/**
 * @brief Validate all configuration components
 *
 * @param options Validation options
 * @param result Output validation result
 * @return 0 on success, -1 on error
 */
int dsmil_validate_all(const dsmil_validation_options_t *options,
                       dsmil_validation_result_t *result);

/**
 * @brief Validate mission profile configuration
 *
 * @param profile_path Path to mission profile JSON file
 * @param result Output validation result
 * @return 0 on success, -1 on error
 */
int dsmil_validate_mission_profiles(const char *profile_path,
                                     dsmil_validation_result_t *result);

/**
 * @brief Validate path configuration
 *
 * Checks that all configured paths exist and are accessible.
 *
 * @param result Output validation result
 * @return 0 on success, -1 on error
 */
int dsmil_validate_paths(dsmil_validation_result_t *result);

/**
 * @brief Validate truststore configuration
 *
 * Checks certificate chains, revocation lists, and key integrity.
 *
 * @param truststore_dir Truststore directory path
 * @param result Output validation result
 * @return 0 on success, -1 on error
 */
int dsmil_validate_truststore(const char *truststore_dir,
                               dsmil_validation_result_t *result);

/**
 * @brief Validate classification configuration
 *
 * Checks cross-domain gateway configurations and classification consistency.
 *
 * @param result Output validation result
 * @return 0 on success, -1 on error
 */
int dsmil_validate_classification(dsmil_validation_result_t *result);

/**
 * @brief Fix common configuration issues automatically
 *
 * @param options Validation options
 * @return Number of issues fixed, -1 on error
 */
int dsmil_auto_fix_config(const dsmil_validation_options_t *options);

/**
 * @brief Generate health report
 *
 * @param output_path Path to output JSON report
 * @param options Validation options
 * @return 0 on success, -1 on error
 */
int dsmil_generate_health_report(const char *output_path,
                                 const dsmil_validation_options_t *options);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_CONFIG_VALIDATOR_H */
