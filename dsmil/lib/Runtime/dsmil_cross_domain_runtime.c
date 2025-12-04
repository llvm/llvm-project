/**
 * @file dsmil_cross_domain_runtime.c
 * @brief DSMIL Cross-Domain Security Runtime (v1.5)
 *
 * Runtime support for DoD classification-aware cross-domain security.
 * Implements guards, validation, and audit logging for classification
 * boundary transitions.
 *
 * Features:
 * - Cross-domain guard validation
 * - Classification downgrade authorization
 * - Audit logging to Layer 62 (Forensics)
 * - Network-based classification enforcement
 *
 * Networks:
 * - NIP (UNCLASSIFIED)
 * - SIPRNET (SECRET)
 * - JWICS (TOP SECRET/SCI)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

// Classification levels (must match compiler enum)
typedef enum {
    DSMIL_CLASS_U = 0,      // UNCLASSIFIED
    DSMIL_CLASS_C = 1,      // CONFIDENTIAL
    DSMIL_CLASS_S = 2,      // SECRET (SIPRNET)
    DSMIL_CLASS_TS = 3,     // TOP SECRET
    DSMIL_CLASS_TS_SCI = 4, // TOP SECRET/SCI (JWICS)
    DSMIL_CLASS_UNKNOWN = 99
} dsmil_classification_t;

// Cross-domain guard policies
typedef enum {
    DSMIL_GUARD_MANUAL_REVIEW,      // Human review required
    DSMIL_GUARD_AUTO_SANITIZE,      // AI-assisted sanitization
    DSMIL_GUARD_REJECT,              // Always reject
    DSMIL_GUARD_AUDIT_ONLY           // Allow but audit
} dsmil_guard_policy_t;

// Guard context (global state)
static struct {
    bool initialized;
    FILE *audit_log;
    uint64_t transition_count;
    uint64_t violation_count;
    dsmil_classification_t current_network_level;
} g_guard_ctx = {0};

/**
 * @brief Parse classification string to enum
 */
static dsmil_classification_t parse_classification(const char *level) {
    if (!level) return DSMIL_CLASS_UNKNOWN;

    if (strcmp(level, "U") == 0 || strcmp(level, "UNCLASSIFIED") == 0)
        return DSMIL_CLASS_U;
    if (strcmp(level, "C") == 0 || strcmp(level, "CONFIDENTIAL") == 0)
        return DSMIL_CLASS_C;
    if (strcmp(level, "S") == 0 || strcmp(level, "SECRET") == 0)
        return DSMIL_CLASS_S;
    if (strcmp(level, "TS") == 0 || strcmp(level, "TOP_SECRET") == 0)
        return DSMIL_CLASS_TS;
    if (strcmp(level, "TS/SCI") == 0 || strcmp(level, "TS_SCI") == 0)
        return DSMIL_CLASS_TS_SCI;

    return DSMIL_CLASS_UNKNOWN;
}

/**
 * @brief Convert classification enum to string
 */
static const char* classification_to_string(dsmil_classification_t level) {
    switch (level) {
        case DSMIL_CLASS_U: return "U";
        case DSMIL_CLASS_C: return "C";
        case DSMIL_CLASS_S: return "S";
        case DSMIL_CLASS_TS: return "TS";
        case DSMIL_CLASS_TS_SCI: return "TS/SCI";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Initialize cross-domain guard subsystem
 *
 * @param network_classification Current network classification (e.g., "S" for SIPRNET)
 * @return 0 on success, negative on error
 */
int dsmil_cross_domain_init(const char *network_classification) {
    if (g_guard_ctx.initialized) {
        return 0;  // Already initialized
    }

    // Parse network classification
    g_guard_ctx.current_network_level = parse_classification(network_classification);

    if (g_guard_ctx.current_network_level == DSMIL_CLASS_UNKNOWN) {
        fprintf(stderr, "ERROR: Invalid network classification: %s\n",
                network_classification);
        return -1;
    }

    // Open audit log for cross-domain transitions (Layer 62 Forensics)
    const char *log_path = getenv("DSMIL_CROSS_DOMAIN_LOG");
    if (!log_path) {
        log_path = "/var/log/dsmil/cross_domain_audit.log";
    }

    g_guard_ctx.audit_log = fopen(log_path, "a");
    if (!g_guard_ctx.audit_log) {
        fprintf(stderr, "WARNING: Could not open cross-domain audit log: %s\n",
                log_path);
        // Continue without logging (for testing)
        g_guard_ctx.audit_log = stderr;
    }

    g_guard_ctx.initialized = true;
    g_guard_ctx.transition_count = 0;
    g_guard_ctx.violation_count = 0;

    fprintf(g_guard_ctx.audit_log,
            "[INIT] Cross-domain guard initialized, network=%s\n",
            network_classification);
    fflush(g_guard_ctx.audit_log);

    return 0;
}

/**
 * @brief Shutdown cross-domain guard subsystem
 */
void dsmil_cross_domain_shutdown(void) {
    if (!g_guard_ctx.initialized) {
        return;
    }

    fprintf(g_guard_ctx.audit_log,
            "[SHUTDOWN] Transitions: %lu, Violations: %lu\n",
            g_guard_ctx.transition_count,
            g_guard_ctx.violation_count);

    if (g_guard_ctx.audit_log != stderr) {
        fclose(g_guard_ctx.audit_log);
    }

    g_guard_ctx.initialized = false;
}

/**
 * @brief Runtime cross-domain guard
 *
 * Validates cross-domain data transition and applies guard policy.
 * All transitions logged to Layer 62 (Forensics).
 *
 * @param data Data being transferred
 * @param length Length of data
 * @param from_level Source classification level
 * @param to_level Destination classification level
 * @param guard_policy Policy to apply
 * @return 0 if allowed, negative if rejected
 */
int dsmil_cross_domain_guard(const void *data,
                               size_t length,
                               const char *from_level,
                               const char *to_level,
                               const char *guard_policy) {
    if (!g_guard_ctx.initialized) {
        dsmil_cross_domain_init("U");  // Default to UNCLASSIFIED
    }

    dsmil_classification_t from = parse_classification(from_level);
    dsmil_classification_t to = parse_classification(to_level);

    g_guard_ctx.transition_count++;

    // Get timestamp
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                             (uint64_t)ts.tv_nsec;

    // Log transition
    fprintf(g_guard_ctx.audit_log,
            "[TRANSITION] ts=%lu from=%s to=%s bytes=%zu policy=%s\n",
            timestamp_ns,
            classification_to_string(from),
            classification_to_string(to),
            length,
            guard_policy ? guard_policy : "none");
    fflush(g_guard_ctx.audit_log);

    // Validate transition
    if (from == DSMIL_CLASS_UNKNOWN || to == DSMIL_CLASS_UNKNOWN) {
        fprintf(g_guard_ctx.audit_log,
                "[VIOLATION] Unknown classification level\n");
        fflush(g_guard_ctx.audit_log);
        g_guard_ctx.violation_count++;
        return -1;
    }

    // Higher→Lower: requires explicit guard policy
    if (from > to) {
        if (!guard_policy || strcmp(guard_policy, "manual_review") != 0) {
            fprintf(g_guard_ctx.audit_log,
                    "[VIOLATION] Downgrade requires manual_review policy\n");
            fflush(g_guard_ctx.audit_log);
            g_guard_ctx.violation_count++;
            return -1;
        }

        // In production, this would trigger manual review workflow
        // For now, log and allow
        fprintf(g_guard_ctx.audit_log,
                "[DOWNGRADE] Manual review required (simulated approval)\n");
        fflush(g_guard_ctx.audit_log);
    }

    // Lower→Higher: generally safe (upgrade)
    if (from < to) {
        fprintf(g_guard_ctx.audit_log,
                "[UPGRADE] Classification upgrade (safe)\n");
        fflush(g_guard_ctx.audit_log);
    }

    // Network boundary check
    if (to > g_guard_ctx.current_network_level) {
        fprintf(g_guard_ctx.audit_log,
                "[VIOLATION] Target classification exceeds network level\n");
        fflush(g_guard_ctx.audit_log);
        g_guard_ctx.violation_count++;
        return -1;
    }

    return 0;  // Allowed
}

/**
 * @brief Check if classification downgrade is authorized
 *
 * @param from_level Source classification
 * @param to_level Destination classification
 * @param authority Authorization authority (e.g., officer name, ML-DSA-87 signature)
 * @return true if authorized, false otherwise
 */
bool dsmil_classification_can_downgrade(const char *from_level,
                                         const char *to_level,
                                         const char *authority) {
    if (!g_guard_ctx.initialized) {
        dsmil_cross_domain_init("U");
    }

    dsmil_classification_t from = parse_classification(from_level);
    dsmil_classification_t to = parse_classification(to_level);

    if (from <= to) {
        return true;  // Not a downgrade
    }

    // Check authorization authority
    // In production, this would verify ML-DSA-87 signature
    if (!authority || strlen(authority) == 0) {
        return false;  // No authority provided
    }

    // Simulate authorization check
    fprintf(g_guard_ctx.audit_log,
            "[AUTH_CHECK] Downgrade %s→%s authorized by %s\n",
            classification_to_string(from),
            classification_to_string(to),
            authority);
    fflush(g_guard_ctx.audit_log);

    return true;  // Simplified: always authorized if authority provided
}

/**
 * @brief Get current network classification level
 *
 * @return Classification level string (e.g., "S" for SIPRNET)
 */
const char* dsmil_get_network_classification(void) {
    if (!g_guard_ctx.initialized) {
        return "UNKNOWN";
    }
    return classification_to_string(g_guard_ctx.current_network_level);
}

/**
 * @brief Validate that function execution is authorized for current classification
 *
 * @param function_name Function name
 * @param required_level Required classification level
 * @return 0 if authorized, negative otherwise
 */
int dsmil_validate_function_classification(const char *function_name,
                                             const char *required_level) {
    if (!g_guard_ctx.initialized) {
        dsmil_cross_domain_init("U");
    }

    dsmil_classification_t required = parse_classification(required_level);

    if (required > g_guard_ctx.current_network_level) {
        fprintf(g_guard_ctx.audit_log,
                "[VIOLATION] Function %s requires %s but network is %s\n",
                function_name,
                required_level,
                classification_to_string(g_guard_ctx.current_network_level));
        fflush(g_guard_ctx.audit_log);
        g_guard_ctx.violation_count++;
        return -1;
    }

    return 0;
}

/**
 * @brief Get cross-domain guard statistics
 *
 * @param total_transitions Output: total number of cross-domain transitions
 * @param violations Output: number of violations detected
 */
void dsmil_cross_domain_stats(uint64_t *total_transitions,
                                uint64_t *violations) {
    if (!g_guard_ctx.initialized) {
        *total_transitions = 0;
        *violations = 0;
        return;
    }

    *total_transitions = g_guard_ctx.transition_count;
    *violations = g_guard_ctx.violation_count;
}
