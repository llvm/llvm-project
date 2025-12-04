/**
 * @file dsmil_blue_red_runtime.c
 * @brief DSLLVM Blue vs Red Runtime Support (v1.4)
 *
 * Runtime support for blue/red build simulation and adversarial testing.
 * Red builds include extra instrumentation to simulate attack scenarios.
 *
 * RED BUILDS ARE FOR TESTING ONLY - NEVER DEPLOY TO PRODUCTION
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

/**
 * Red build flag (set at runtime by loader)
 */
static int g_is_red_build = 0;

/**
 * Scenario configuration (set via environment or config file)
 */
static char g_active_scenarios[256] = {0};

/**
 * Red team log file
 */
static FILE *g_red_log_file = NULL;

/**
 * Initialize blue/red runtime
 *
 * @param is_red_build 1 if red build, 0 if blue build
 * @return 0 on success, -1 on error
 *
 * Must be called during process initialization.
 */
int dsmil_blue_red_init(int is_red_build) {
    g_is_red_build = is_red_build;

    if (g_is_red_build) {
        // RED BUILD WARNING
        fprintf(stderr, "\n");
        fprintf(stderr, "========================================\n");
        fprintf(stderr, "WARNING: DSMIL RED TEAM BUILD\n");
        fprintf(stderr, "FOR TESTING ONLY\n");
        fprintf(stderr, "NEVER DEPLOY TO PRODUCTION\n");
        fprintf(stderr, "========================================\n");
        fprintf(stderr, "\n");

        // Open red team log file
        const char *log_path = getenv("DSMIL_RED_LOG");
        if (!log_path) {
            log_path = "/tmp/dsmil-red.log";
        }

        g_red_log_file = fopen(log_path, "a");
        if (!g_red_log_file) {
            fprintf(stderr, "ERROR: Failed to open red log: %s\n", log_path);
            return -1;
        }

        fprintf(g_red_log_file, "\n=== RED BUILD SESSION START ===\n");
        fprintf(g_red_log_file, "Timestamp: %ld\n", (long)time(NULL));
        fflush(g_red_log_file);

        // Load active scenarios from environment
        const char *scenarios = getenv("DSMIL_RED_SCENARIOS");
        if (scenarios) {
            strncpy(g_active_scenarios, scenarios, sizeof(g_active_scenarios) - 1);
            g_active_scenarios[sizeof(g_active_scenarios) - 1] = '\0';
            fprintf(g_red_log_file, "Active scenarios: %s\n", g_active_scenarios);
            fflush(g_red_log_file);
        }
    }

    return 0;
}

/**
 * Shutdown blue/red runtime
 *
 * Flushes logs and releases resources.
 */
void dsmil_blue_red_shutdown(void) {
    if (g_is_red_build && g_red_log_file) {
        fprintf(g_red_log_file, "=== RED BUILD SESSION END ===\n\n");
        fclose(g_red_log_file);
        g_red_log_file = NULL;
    }
}

/**
 * Check if current build is red team build
 *
 * @return 1 if red build, 0 if blue build
 */
int dsmil_is_red_build(void) {
    return g_is_red_build;
}

/**
 * Log red team event
 *
 * @param hook_name Hook identifier
 * @param function_name Function name
 *
 * Logs instrumentation point execution. Only active in red builds.
 */
void dsmil_red_log(const char *hook_name, const char *function_name) {
    if (!g_is_red_build || !g_red_log_file)
        return;

    time_t now = time(NULL);
    fprintf(g_red_log_file, "[%ld] RED_HOOK: %s in %s\n",
            (long)now, hook_name, function_name);
    fflush(g_red_log_file);
}

/**
 * Log red team event with details
 *
 * @param hook_name Hook identifier
 * @param function_name Function name
 * @param details Additional details (format string)
 * @param ... Format arguments
 */
void dsmil_red_log_detailed(const char *hook_name,
                            const char *function_name,
                            const char *details, ...) {
    if (!g_is_red_build || !g_red_log_file)
        return;

    time_t now = time(NULL);
    fprintf(g_red_log_file, "[%ld] RED_HOOK: %s in %s - ",
            (long)now, hook_name, function_name);

    va_list args;
    va_start(args, details);
    vfprintf(g_red_log_file, details, args);
    va_end(args);

    fprintf(g_red_log_file, "\n");
    fflush(g_red_log_file);
}

/**
 * Check if red team scenario is active
 *
 * @param scenario_name Scenario identifier
 * @return 1 if scenario is active, 0 otherwise
 *
 * Scenarios are controlled via DSMIL_RED_SCENARIOS environment variable:
 * - "all": All scenarios active
 * - "scenario1,scenario2": Specific scenarios
 * - empty: No scenarios (normal execution)
 *
 * Example:
 *   export DSMIL_RED_SCENARIOS="bypass_validation,trigger_overflow"
 */
int dsmil_red_scenario(const char *scenario_name) {
    if (!g_is_red_build)
        return 0;

    // If no scenarios configured, return 0 (normal execution)
    if (g_active_scenarios[0] == '\0')
        return 0;

    // Check for "all" wildcard
    if (strcmp(g_active_scenarios, "all") == 0)
        return 1;

    // Check if scenario is in comma-separated list
    char *scenarios_copy = strdup(g_active_scenarios);
    char *token = strtok(scenarios_copy, ",");

    while (token != NULL) {
        // Trim whitespace
        while (*token == ' ') token++;
        char *end = token + strlen(token) - 1;
        while (end > token && *end == ' ') end--;
        *(end + 1) = '\0';

        if (strcmp(token, scenario_name) == 0) {
            free(scenarios_copy);

            if (g_red_log_file) {
                fprintf(g_red_log_file, "[%ld] SCENARIO_ACTIVE: %s\n",
                        (long)time(NULL), scenario_name);
                fflush(g_red_log_file);
            }

            return 1;
        }

        token = strtok(NULL, ",");
    }

    free(scenarios_copy);
    return 0;
}

/**
 * Log attack surface entry
 *
 * @param function_name Function name
 * @param untrusted_data Pointer to untrusted data (for logging size/type)
 * @param data_size Size of untrusted data
 *
 * Logs entry to attack surface function. Used for blast radius analysis.
 */
void dsmil_red_attack_surface_entry(const char *function_name,
                                    const void *untrusted_data,
                                    size_t data_size) {
    if (!g_is_red_build || !g_red_log_file)
        return;

    fprintf(g_red_log_file, "[%ld] ATTACK_SURFACE: %s (data_size=%zu)\n",
            (long)time(NULL), function_name, data_size);
    fflush(g_red_log_file);
}

/**
 * Log vulnerability injection trigger
 *
 * @param vuln_type Vulnerability type
 * @param function_name Function name
 * @param details Additional details
 */
void dsmil_red_vuln_inject_log(const char *vuln_type,
                               const char *function_name,
                               const char *details) {
    if (!g_is_red_build || !g_red_log_file)
        return;

    fprintf(g_red_log_file, "[%ld] VULN_INJECT: %s in %s - %s\n",
            (long)time(NULL), vuln_type, function_name, details);
    fflush(g_red_log_file);
}

/**
 * Log blast radius event
 *
 * @param function_name Function name
 * @param event Event type (e.g., "compromised", "escalated")
 * @param details Additional details
 */
void dsmil_red_blast_radius_event(const char *function_name,
                                  const char *event,
                                  const char *details) {
    if (!g_is_red_build || !g_red_log_file)
        return;

    fprintf(g_red_log_file, "[%ld] BLAST_RADIUS: %s - %s: %s\n",
            (long)time(NULL), function_name, event, details);
    fflush(g_red_log_file);
}

/**
 * Get red build statistics
 *
 * @param red_hooks_triggered Output: number of red hooks triggered
 * @param scenarios_activated Output: number of scenarios activated
 * @param attack_surfaces_hit Output: number of attack surfaces entered
 * @return 0 on success, -1 on error
 */
int dsmil_red_get_stats(unsigned *red_hooks_triggered,
                        unsigned *scenarios_activated,
                        unsigned *attack_surfaces_hit) {
    // TODO: Implement statistics tracking
    // For now, return zeros

    if (red_hooks_triggered) *red_hooks_triggered = 0;
    if (scenarios_activated) *scenarios_activated = 0;
    if (attack_surfaces_hit) *attack_surfaces_hit = 0;

    return 0;
}

/**
 * Enable/disable red team scenario at runtime
 *
 * @param scenario_name Scenario identifier
 * @param enabled 1 to enable, 0 to disable
 * @return 0 on success, -1 on error
 */
int dsmil_red_set_scenario(const char *scenario_name, int enabled) {
    if (!g_is_red_build)
        return -1;

    // TODO: Implement dynamic scenario control
    // For now, just log the request

    if (g_red_log_file) {
        fprintf(g_red_log_file, "[%ld] SET_SCENARIO: %s = %s\n",
                (long)time(NULL), scenario_name,
                enabled ? "enabled" : "disabled");
        fflush(g_red_log_file);
    }

    return 0;
}

/**
 * Verify blue/red build role
 *
 * @param expected_role "blue" or "red"
 * @return 1 if role matches, 0 otherwise
 *
 * Used by runtime loader to verify build role and reject mismatched binaries.
 */
int dsmil_verify_build_role(const char *expected_role) {
    int is_red = g_is_red_build;
    int expected_red = (strcmp(expected_role, "red") == 0);

    if (is_red != expected_red) {
        fprintf(stderr, "ERROR: Build role mismatch!\n");
        fprintf(stderr, "Expected: %s\n", expected_role);
        fprintf(stderr, "Actual: %s\n", is_red ? "red" : "blue");
        fprintf(stderr, "\nRED BUILDS MUST NOT BE DEPLOYED TO PRODUCTION!\n");
        return 0;
    }

    return 1;
}
