/**
 * @file dsmil_setup.h
 * @brief DSLLVM Setup and Configuration Wizard API
 *
 * Provides functions for interactive setup and configuration of DSLLVM.
 *
 * Version: 1.7.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_SETUP_H
#define DSMIL_SETUP_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_SETUP Setup and Configuration
 * @{
 */

/**
 * @brief Setup options
 */
typedef struct {
    bool interactive;
    bool verify_only;
    bool fix_issues;
    const char *profile_template;
    const char *output_path;
    const char *prefix;
} dsmil_setup_options_t;

/**
 * @brief Run interactive setup wizard
 *
 * @param options Setup options
 * @return 0 on success, -1 on error
 */
int dsmil_setup_wizard(const dsmil_setup_options_t *options);

/**
 * @brief Verify existing installation
 *
 * @param options Setup options
 * @return 0 if valid, -1 if issues found
 */
int dsmil_setup_verify(const dsmil_setup_options_t *options);

/**
 * @brief Fix common installation issues
 *
 * @param options Setup options
 * @return Number of issues fixed, -1 on error
 */
int dsmil_setup_fix(const dsmil_setup_options_t *options);

/**
 * @brief Generate configuration from template
 *
 * @param template_name Template name (e.g., "border_ops", "cyber_defence")
 * @param output_path Output configuration file path
 * @return 0 on success, -1 on error
 */
int dsmil_setup_generate_config(const char *template_name, const char *output_path);

/**
 * @brief Detect DSLLVM installation
 *
 * @param prefix Output prefix path (if found)
 * @param prefix_size Size of prefix buffer
 * @return 0 if found, -1 if not found
 */
int dsmil_setup_detect_installation(char *prefix, size_t prefix_size);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_SETUP_H */
