/**
 * @file dsmil_mpe_runtime.c
 * @brief DSMIL Mission Partner Environment (MPE) Runtime (v1.6.0)
 *
 * Runtime validation for coalition partner access and releasability controls.
 * Implements Mission Partner Environment (MPE) protocol for dynamic coalition
 * operations with NATO, Five Eyes, and other authorized partners.
 *
 * MPE Protocol:
 * - Partner authentication via PKI certificates
 * - Releasability validation (REL NATO, REL FVEY, NOFORN, etc.)
 * - Dynamic coalition membership management
 * - Audit logging of all coalition data sharing
 *
 * Supported Coalitions:
 * - Five Eyes (FVEY): US, UK, CA, AU, NZ
 * - NATO: 32 partner nations
 * - Bilateral partnerships (e.g., REL UK, REL FR)
 * - Mission-specific coalitions
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
#include <time.h>

// Maximum coalition partners per operation
#define MPE_MAX_PARTNERS 32

// Partner authentication
typedef struct {
    char country_code[8];      // ISO 3166-1 alpha-2 (e.g., "US", "UK")
    char organization[64];     // E.g., "US_CENTCOM", "UK_MOD"
    uint8_t cert_hash[32];     // SHA-256 hash of PKI certificate
    bool authenticated;
    uint64_t auth_timestamp_ns;
} dsmil_mpe_partner_t;

// Releasability policy
typedef enum {
    MPE_REL_NOFORN,      // U.S. only
    MPE_REL_FOUO,        // U.S. government only
    MPE_REL_FVEY,        // Five Eyes
    MPE_REL_NATO,        // NATO partners
    MPE_REL_SPECIFIC     // Specific partners
} dsmil_mpe_releasability_t;

// Coalition operation
typedef struct {
    char operation_name[128];
    dsmil_mpe_partner_t partners[MPE_MAX_PARTNERS];
    size_t num_partners;
    dsmil_mpe_releasability_t default_releasability;
    bool active;
} dsmil_mpe_operation_t;

// Global MPE context
static struct {
    bool initialized;
    FILE *mpe_log;

    // Current coalition operation
    dsmil_mpe_operation_t current_op;

    // Statistics
    uint64_t coalition_ops;
    uint64_t data_shared;
    uint64_t access_denied;
    uint64_t releasability_violations;

} g_mpe_ctx = {0};

// Five Eyes partners
static const char *FVEY_PARTNERS[] = {"US", "UK", "CA", "AU", "NZ"};
static const size_t NUM_FVEY = 5;

// NATO partners (32 nations as of 2024)
static const char *NATO_PARTNERS[] = {
    "US", "UK", "CA", "FR", "DE", "IT", "ES", "PL", "NL", "BE",
    "CZ", "GR", "PT", "HU", "RO", "NO", "DK", "BG", "SK", "SI",
    "LT", "LV", "EE", "HR", "AL", "IS", "LU", "ME", "MK", "TR",
    "FI", "SE"
};
static const size_t NUM_NATO = 32;

/**
 * @brief Initialize MPE subsystem
 *
 * @param operation_name Coalition operation name
 * @param default_rel Default releasability policy
 * @return 0 on success, negative on error
 */
int dsmil_mpe_init(const char *operation_name,
                   dsmil_mpe_releasability_t default_rel) {
    if (g_mpe_ctx.initialized) {
        return 0;
    }

    // Open MPE audit log
    const char *log_path = getenv("DSMIL_MPE_LOG");
    if (!log_path) {
        log_path = "/var/log/dsmil/mpe_coalition.log";
    }

    g_mpe_ctx.mpe_log = fopen(log_path, "a");
    if (!g_mpe_ctx.mpe_log) {
        g_mpe_ctx.mpe_log = stderr;
    }

    // Initialize coalition operation
    snprintf(g_mpe_ctx.current_op.operation_name,
             sizeof(g_mpe_ctx.current_op.operation_name),
             "%s", operation_name);
    g_mpe_ctx.current_op.default_releasability = default_rel;
    g_mpe_ctx.current_op.num_partners = 0;
    g_mpe_ctx.current_op.active = true;

    g_mpe_ctx.initialized = true;
    g_mpe_ctx.coalition_ops = 0;
    g_mpe_ctx.data_shared = 0;
    g_mpe_ctx.access_denied = 0;
    g_mpe_ctx.releasability_violations = 0;

    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_INIT] Operation: %s, Releasability: %d\n",
            operation_name, default_rel);
    fflush(g_mpe_ctx.mpe_log);

    return 0;
}

/**
 * @brief Add coalition partner to current operation
 *
 * @param country_code Partner country code (ISO 3166-1 alpha-2)
 * @param organization Partner organization
 * @param cert_hash SHA-256 hash of partner's PKI certificate (32 bytes)
 * @return 0 on success, negative on error
 */
int dsmil_mpe_add_partner(const char *country_code,
                          const char *organization,
                          const uint8_t *cert_hash) {
    if (!g_mpe_ctx.initialized) {
        dsmil_mpe_init("default_coalition", MPE_REL_NATO);
    }

    if (g_mpe_ctx.current_op.num_partners >= MPE_MAX_PARTNERS) {
        fprintf(g_mpe_ctx.mpe_log,
                "[MPE_ERROR] Maximum partners (%d) exceeded\n", MPE_MAX_PARTNERS);
        return -1;
    }

    // Add partner
    dsmil_mpe_partner_t *partner =
        &g_mpe_ctx.current_op.partners[g_mpe_ctx.current_op.num_partners++];

    snprintf(partner->country_code, sizeof(partner->country_code),
             "%s", country_code);
    snprintf(partner->organization, sizeof(partner->organization),
             "%s", organization);
    memcpy(partner->cert_hash, cert_hash, 32);
    partner->authenticated = true;  // Simplified - production would verify cert

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    partner->auth_timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                                 (uint64_t)ts.tv_nsec;

    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_PARTNER_ADD] Country: %s, Org: %s\n",
            country_code, organization);
    fflush(g_mpe_ctx.mpe_log);

    return 0;
}

/**
 * @brief Check if partner is in coalition group
 *
 * @param country_code Partner country code
 * @param coalition Coalition group (FVEY, NATO, etc.)
 * @return true if partner is in coalition, false otherwise
 */
static bool is_in_coalition(const char *country_code, const char *coalition) {
    if (strcmp(coalition, "FVEY") == 0) {
        for (size_t i = 0; i < NUM_FVEY; i++) {
            if (strcmp(country_code, FVEY_PARTNERS[i]) == 0)
                return true;
        }
        return false;
    }

    if (strcmp(coalition, "NATO") == 0) {
        for (size_t i = 0; i < NUM_NATO; i++) {
            if (strcmp(country_code, NATO_PARTNERS[i]) == 0)
                return true;
        }
        return false;
    }

    return false;
}

/**
 * @brief Validate partner access to data
 *
 * @param country_code Partner requesting access
 * @param releasability Data releasability marking
 * @return true if access granted, false if denied
 */
bool dsmil_mpe_validate_access(const char *country_code,
                                const char *releasability) {
    if (!g_mpe_ctx.initialized) {
        return false;
    }

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL +
                            (uint64_t)ts.tv_nsec;

    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_ACCESS_CHECK] Country: %s, Rel: %s, ts: %lu\n",
            country_code, releasability, timestamp_ns);
    fflush(g_mpe_ctx.mpe_log);

    // NOFORN: Only U.S. access
    if (strcmp(releasability, "NOFORN") == 0) {
        if (strcmp(country_code, "US") == 0) {
            g_mpe_ctx.data_shared++;
            fprintf(g_mpe_ctx.mpe_log, "[MPE_GRANTED] NOFORN: U.S. access\n");
            fflush(g_mpe_ctx.mpe_log);
            return true;
        } else {
            g_mpe_ctx.access_denied++;
            fprintf(g_mpe_ctx.mpe_log,
                    "[MPE_DENIED] NOFORN data requested by foreign partner %s\n",
                    country_code);
            fflush(g_mpe_ctx.mpe_log);
            return false;
        }
    }

    // FOUO: U.S. government only
    if (strcmp(releasability, "FOUO") == 0) {
        if (strcmp(country_code, "US") == 0) {
            g_mpe_ctx.data_shared++;
            return true;
        } else {
            g_mpe_ctx.access_denied++;
            return false;
        }
    }

    // REL FVEY: Five Eyes only
    if (strcmp(releasability, "REL FVEY") == 0 ||
        strcmp(releasability, "REL_FVEY") == 0) {
        if (is_in_coalition(country_code, "FVEY")) {
            g_mpe_ctx.data_shared++;
            fprintf(g_mpe_ctx.mpe_log,
                    "[MPE_GRANTED] FVEY access for %s\n", country_code);
            fflush(g_mpe_ctx.mpe_log);
            return true;
        } else {
            g_mpe_ctx.access_denied++;
            fprintf(g_mpe_ctx.mpe_log,
                    "[MPE_DENIED] FVEY data requested by non-FVEY partner %s\n",
                    country_code);
            fflush(g_mpe_ctx.mpe_log);
            return false;
        }
    }

    // REL NATO: NATO partners
    if (strcmp(releasability, "REL NATO") == 0 ||
        strcmp(releasability, "REL_NATO") == 0) {
        if (is_in_coalition(country_code, "NATO")) {
            g_mpe_ctx.data_shared++;
            fprintf(g_mpe_ctx.mpe_log,
                    "[MPE_GRANTED] NATO access for %s\n", country_code);
            fflush(g_mpe_ctx.mpe_log);
            return true;
        } else {
            g_mpe_ctx.access_denied++;
            fprintf(g_mpe_ctx.mpe_log,
                    "[MPE_DENIED] NATO data requested by non-NATO partner %s\n",
                    country_code);
            fflush(g_mpe_ctx.mpe_log);
            return false;
        }
    }

    // REL [specific countries]
    if (strncmp(releasability, "REL ", 4) == 0) {
        const char *countries = releasability + 4;
        char countries_copy[256];
        snprintf(countries_copy, sizeof(countries_copy), "%s", countries);

        // Parse comma-separated country codes
        char *token = strtok(countries_copy, ",");
        while (token) {
            // Trim whitespace
            while (*token == ' ') token++;
            char *end = token + strlen(token) - 1;
            while (end > token && *end == ' ') *end-- = '\0';

            if (strcmp(token, country_code) == 0) {
                g_mpe_ctx.data_shared++;
                fprintf(g_mpe_ctx.mpe_log,
                        "[MPE_GRANTED] Specific release to %s\n", country_code);
                fflush(g_mpe_ctx.mpe_log);
                return true;
            }

            token = strtok(NULL, ",");
        }

        // Country not in authorized list
        g_mpe_ctx.access_denied++;
        fprintf(g_mpe_ctx.mpe_log,
                "[MPE_DENIED] %s not in authorized list: %s\n",
                country_code, releasability);
        fflush(g_mpe_ctx.mpe_log);
        return false;
    }

    // Unknown releasability - deny by default
    g_mpe_ctx.access_denied++;
    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_DENIED] Unknown releasability: %s\n", releasability);
    fflush(g_mpe_ctx.mpe_log);
    return false;
}

/**
 * @brief Check if partner is authenticated in current coalition
 *
 * @param country_code Partner country code
 * @return true if authenticated, false otherwise
 */
bool dsmil_mpe_is_partner_authenticated(const char *country_code) {
    if (!g_mpe_ctx.initialized) {
        return false;
    }

    for (size_t i = 0; i < g_mpe_ctx.current_op.num_partners; i++) {
        dsmil_mpe_partner_t *partner = &g_mpe_ctx.current_op.partners[i];
        if (strcmp(partner->country_code, country_code) == 0) {
            return partner->authenticated;
        }
    }

    return false;
}

/**
 * @brief Share data with coalition partner
 *
 * @param data Data to share
 * @param length Data length
 * @param releasability Releasability marking
 * @param partner_country Target partner country code
 * @return 0 on success, negative on error
 */
int dsmil_mpe_share_data(const void *data, size_t length,
                         const char *releasability,
                         const char *partner_country) {
    if (!g_mpe_ctx.initialized) {
        dsmil_mpe_init("default_coalition", MPE_REL_NATO);
    }

    // Validate partner access
    if (!dsmil_mpe_validate_access(partner_country, releasability)) {
        fprintf(g_mpe_ctx.mpe_log,
                "[MPE_SHARE_DENIED] Access denied for %s (rel: %s)\n",
                partner_country, releasability);
        fflush(g_mpe_ctx.mpe_log);
        g_mpe_ctx.releasability_violations++;
        return -1;
    }

    // Check partner authentication
    if (!dsmil_mpe_is_partner_authenticated(partner_country)) {
        fprintf(g_mpe_ctx.mpe_log,
                "[MPE_SHARE_DENIED] Partner %s not authenticated\n",
                partner_country);
        fflush(g_mpe_ctx.mpe_log);
        return -1;
    }

    // Share data (production would encrypt and transmit)
    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_SHARE] Sharing %zu bytes with %s (rel: %s)\n",
            length, partner_country, releasability);
    fflush(g_mpe_ctx.mpe_log);

    g_mpe_ctx.data_shared++;
    g_mpe_ctx.coalition_ops++;

    (void)data;  // Suppress unused warning

    return 0;
}

/**
 * @brief Get MPE operation status
 *
 * @param op_name Output: operation name
 * @param num_partners Output: number of coalition partners
 * @param active Output: operation active status
 */
void dsmil_mpe_get_status(char *op_name, size_t *num_partners, bool *active) {
    if (!g_mpe_ctx.initialized) {
        *op_name = '\0';
        *num_partners = 0;
        *active = false;
        return;
    }

    snprintf(op_name, 128, "%s", g_mpe_ctx.current_op.operation_name);
    *num_partners = g_mpe_ctx.current_op.num_partners;
    *active = g_mpe_ctx.current_op.active;
}

/**
 * @brief Get MPE statistics
 *
 * @param coalition_ops Output: coalition operations count
 * @param data_shared Output: data shared count
 * @param access_denied Output: access denied count
 * @param violations Output: releasability violations count
 */
void dsmil_mpe_get_stats(uint64_t *coalition_ops, uint64_t *data_shared,
                         uint64_t *access_denied, uint64_t *violations) {
    if (!g_mpe_ctx.initialized) {
        *coalition_ops = 0;
        *data_shared = 0;
        *access_denied = 0;
        *violations = 0;
        return;
    }

    *coalition_ops = g_mpe_ctx.coalition_ops;
    *data_shared = g_mpe_ctx.data_shared;
    *access_denied = g_mpe_ctx.access_denied;
    *violations = g_mpe_ctx.releasability_violations;
}

/**
 * @brief Shutdown MPE subsystem
 */
void dsmil_mpe_shutdown(void) {
    if (!g_mpe_ctx.initialized) {
        return;
    }

    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_SHUTDOWN] Operation: %s, Partners: %zu\n",
            g_mpe_ctx.current_op.operation_name,
            g_mpe_ctx.current_op.num_partners);
    fprintf(g_mpe_ctx.mpe_log,
            "[MPE_SHUTDOWN] CoalitionOps=%lu Shared=%lu Denied=%lu Violations=%lu\n",
            g_mpe_ctx.coalition_ops,
            g_mpe_ctx.data_shared,
            g_mpe_ctx.access_denied,
            g_mpe_ctx.releasability_violations);

    if (g_mpe_ctx.mpe_log != stderr) {
        fclose(g_mpe_ctx.mpe_log);
    }

    g_mpe_ctx.initialized = false;
}
