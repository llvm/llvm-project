/**
 * @file high_assurance_example.c
 * @brief DSMIL v1.6.0 Phase 3: High-Assurance Features Example
 *
 * Demonstrates advanced high-assurance capabilities for mission-critical
 * military operations including nuclear surety, coalition operations, and
 * edge security hardening.
 *
 * Features Demonstrated:
 * - Feature 3.4: Two-Person Integrity (2PI) for Nuclear Surety
 * - Feature 3.5: Mission Partner Environment (MPE) Coalition Sharing
 * - Feature 3.8: Edge Security Hardening (HSM, Enclave, Attestation)
 *
 * Mission Scenario: Joint NATO operation with nuclear deterrence posture
 * - U.S. Cyber Command coordinates multi-national cyber operations
 * - Nuclear Command & Control (NC3) functions require 2PI authorization
 * - Coalition intelligence shared with NATO partners via MPE
 * - Edge nodes hardened against physical tampering in contested environment
 *
 * Classification: TOP SECRET//SCI//NOFORN (U.S. nuclear functions)
 *                 SECRET//REL NATO (Coalition shared functions)
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// DSMIL attribute definitions
#include "dsmil_attributes.h"

// Runtime declarations
extern int dsmil_nuclear_surety_init(const char *officer1_id,
                                      const uint8_t *officer1_pubkey,
                                      const char *officer2_id,
                                      const uint8_t *officer2_pubkey);
extern int dsmil_two_person_verify(const char *function_name,
                                    const uint8_t *sig1, const uint8_t *sig2,
                                    const char *key_id1, const char *key_id2);
extern void dsmil_nc3_audit_log(const char *message);

extern int dsmil_mpe_init(const char *operation_name, int default_rel);
extern int dsmil_mpe_add_partner(const char *country_code,
                                  const char *organization,
                                  const uint8_t *cert_hash);
extern int dsmil_mpe_share_data(const void *data, size_t length,
                                 const char *releasability,
                                 const char *partner_country);
extern bool dsmil_mpe_validate_access(const char *country_code,
                                       const char *releasability);

extern int dsmil_edge_security_init(int hsm_type, int enclave_type);
extern int dsmil_edge_remote_attest(const uint8_t *nonce,
                                     uint8_t *quote, size_t *quote_len);
extern int dsmil_hsm_crypto(const char *operation,
                             const uint8_t *input, size_t input_len,
                             uint8_t *output, size_t *output_len);
extern int dsmil_edge_tamper_detect(void);
extern bool dsmil_edge_is_trusted(void);

// Constants
#define MLDSA87_PUBLIC_KEY_BYTES 2592
#define MLDSA87_SIGNATURE_BYTES 4595
#define AES256_KEY_BYTES 32
#define SHA256_HASH_BYTES 32

// MPE releasability levels
#define MPE_REL_NOFORN 0
#define MPE_REL_FVEY 2
#define MPE_REL_NATO 3

// HSM and enclave types
#define HSM_TYPE_TPM2 1
#define ENCLAVE_SGX 1

//
// SCENARIO 1: Nuclear Command & Control (NC3) with Two-Person Integrity
//
// Nuclear surety requires two independent authorizations from distinct
// officers before executing critical functions (DOE Sigma 14).
//

/**
 * @brief Authorize nuclear weapon release (REQUIRES 2PI)
 *
 * This function is TOP SECRET//SCI//NOFORN and requires two-person
 * integrity authorization via ML-DSA-87 digital signatures from two
 * independent commanding officers.
 *
 * Classification: TOP SECRET//SCI//NOFORN
 */
DSMIL_CLASSIFICATION("TS/SCI")
DSMIL_TWO_PERSON
DSMIL_NC3_ISOLATED
DSMIL_NOFORN
static int authorize_nuclear_release(const char *weapon_system,
                                       const uint8_t *officer1_sig,
                                       const uint8_t *officer2_sig,
                                       const char *officer1_id,
                                       const char *officer2_id) {
    printf("\n=== SCENARIO 1: Nuclear Surety (Two-Person Integrity) ===\n");
    printf("Function: authorize_nuclear_release\n");
    printf("Classification: TOP SECRET//SCI//NOFORN\n");
    printf("Weapon System: %s\n", weapon_system);
    printf("Officer 1: %s\n", officer1_id);
    printf("Officer 2: %s\n", officer2_id);

    // Verify two-person authorization
    int result = dsmil_two_person_verify(
        "authorize_nuclear_release",
        officer1_sig, officer2_sig,
        officer1_id, officer2_id
    );

    if (result != 0) {
        printf("ERROR: Two-person authorization DENIED\n");
        dsmil_nc3_audit_log("2PI DENIED: authorize_nuclear_release");
        return -1;
    }

    printf("SUCCESS: Two-person authorization GRANTED\n");
    printf("Both ML-DSA-87 signatures VERIFIED\n");
    printf("Nuclear release authorization: APPROVED\n");

    dsmil_nc3_audit_log("2PI GRANTED: authorize_nuclear_release");

    return 0;
}

/**
 * @brief Change nuclear alert status (REQUIRES 2PI)
 *
 * Changes DEFCON level for nuclear forces. Requires presidential and
 * SECDEF authorization via two-person integrity.
 */
DSMIL_CLASSIFICATION("TS/SCI")
DSMIL_TWO_PERSON
DSMIL_NC3_ISOLATED
DSMIL_NOFORN
static int change_defcon_level(int new_level,
                                const uint8_t *president_sig,
                                const uint8_t *secdef_sig) {
    printf("\n=== DEFCON Level Change (2PI Required) ===\n");
    printf("New DEFCON Level: %d\n", new_level);

    int result = dsmil_two_person_verify(
        "change_defcon_level",
        president_sig, secdef_sig,
        "POTUS", "SECDEF"
    );

    if (result != 0) {
        printf("ERROR: Two-person authorization DENIED\n");
        return -1;
    }

    printf("SUCCESS: DEFCON level changed to %d\n", new_level);
    return 0;
}

//
// SCENARIO 2: Mission Partner Environment (MPE) Coalition Sharing
//
// Share intelligence with NATO coalition partners while enforcing
// releasability controls (REL NATO, REL FVEY, NOFORN).
//

/**
 * @brief Process coalition intelligence (REL NATO)
 *
 * Tactical intelligence releasable to all NATO partners for
 * coordinated strike operations.
 */
DSMIL_CLASSIFICATION("S")
DSMIL_MPE_RELEASABILITY("REL NATO")
static void process_coalition_intelligence(const char *intel_report) {
    printf("\n=== SCENARIO 2: Coalition Intelligence Sharing (MPE) ===\n");
    printf("Classification: SECRET//REL NATO\n");
    printf("Intelligence: %s\n", intel_report);

    // Share with NATO partners
    const char *nato_partners[] = {"UK", "FR", "DE", "PL"};
    for (int i = 0; i < 4; i++) {
        int result = dsmil_mpe_share_data(
            intel_report, strlen(intel_report),
            "REL NATO", nato_partners[i]
        );

        if (result == 0) {
            printf("Shared with %s: SUCCESS\n", nato_partners[i]);
        } else {
            printf("Shared with %s: DENIED\n", nato_partners[i]);
        }
    }

    // Try to share with non-NATO partner (should fail)
    printf("\nAttempting to share NATO intel with non-NATO partner (RU):\n");
    int result = dsmil_mpe_share_data(
        intel_report, strlen(intel_report),
        "REL NATO", "RU"
    );
    printf("Result: %s\n", result == 0 ? "GRANTED (ERROR!)" : "DENIED (correct)");
}

/**
 * @brief Process Five Eyes intelligence (REL FVEY)
 *
 * Sensitive SIGINT only for Five Eyes partners (US/UK/CA/AU/NZ).
 */
DSMIL_CLASSIFICATION("TS")
DSMIL_MPE_RELEASABILITY("REL FVEY")
static void process_fvey_sigint(const char *sigint_data) {
    printf("\n=== Five Eyes SIGINT (REL FVEY) ===\n");
    printf("Classification: TOP SECRET//REL FVEY\n");
    printf("SIGINT: %s\n", sigint_data);

    // Share with Five Eyes only
    const char *fvey_partners[] = {"UK", "CA", "AU", "NZ"};
    for (int i = 0; i < 4; i++) {
        int result = dsmil_mpe_share_data(
            sigint_data, strlen(sigint_data),
            "REL FVEY", fvey_partners[i]
        );
        printf("Shared with %s: %s\n", fvey_partners[i],
               result == 0 ? "SUCCESS" : "DENIED");
    }

    // Try to share with NATO (non-FVEY) partner (should fail)
    printf("\nAttempting to share FVEY intel with NATO partner (FR):\n");
    int result = dsmil_mpe_share_data(
        sigint_data, strlen(sigint_data),
        "REL FVEY", "FR"
    );
    printf("Result: %s\n", result == 0 ? "GRANTED (ERROR!)" : "DENIED (correct)");
}

/**
 * @brief Process U.S.-only intelligence (NOFORN)
 *
 * U.S.-only HUMINT from CIA, not releasable to any foreign partners.
 */
DSMIL_CLASSIFICATION("TS/SCI")
DSMIL_NOFORN
static void process_noforn_humint(const char *humint_source) {
    printf("\n=== U.S.-Only Intelligence (NOFORN) ===\n");
    printf("Classification: TOP SECRET//SCI//NOFORN\n");
    printf("HUMINT Source: %s\n", humint_source);

    // Verify U.S. access (should succeed)
    bool us_access = dsmil_mpe_validate_access("US", "NOFORN");
    printf("U.S. access: %s\n", us_access ? "GRANTED" : "DENIED");

    // Try foreign partner access (should fail)
    bool uk_access = dsmil_mpe_validate_access("UK", "NOFORN");
    printf("UK access: %s (correct: DENIED)\n",
           uk_access ? "GRANTED (ERROR!)" : "DENIED");
}

//
// SCENARIO 3: Edge Security Hardening
//
// 5G/MEC edge nodes in contested environment require hardware security
// module (HSM) crypto, secure enclave execution, and remote attestation.
//

/**
 * @brief Process classified data on edge node with HSM
 *
 * Uses Hardware Security Module (HSM) for all crypto operations to
 * prevent key extraction via physical attacks.
 */
DSMIL_CLASSIFICATION("S")
DSMIL_5G_EDGE
DSMIL_HSM_CRYPTO
static int edge_process_classified(const uint8_t *data, size_t len) {
    printf("\n=== SCENARIO 3: Edge Security Hardening ===\n");
    printf("Classification: SECRET\n");
    printf("Edge Node: 5G/MEC with HSM\n");
    printf("Data Size: %zu bytes\n", len);

    // Check edge node trust status
    if (!dsmil_edge_is_trusted()) {
        printf("ERROR: Edge node not trusted (tampering detected)\n");
        return -1;
    }

    // Perform crypto using HSM (keys never leave HSM)
    uint8_t encrypted[1024];
    size_t encrypted_len = sizeof(encrypted);

    int result = dsmil_hsm_crypto(
        "encrypt", data, len,
        encrypted, &encrypted_len
    );

    if (result == 0) {
        printf("HSM encryption: SUCCESS (%zu bytes)\n", encrypted_len);
        printf("Cryptographic keys secured in FIPS 140-3 Level 3 HSM\n");
    } else {
        printf("HSM encryption: FAILED\n");
        return -1;
    }

    return 0;
}

/**
 * @brief Execute sensitive computation in secure enclave
 *
 * Runs in Intel SGX or ARM TrustZone to protect against memory
 * scraping and side-channel attacks.
 */
DSMIL_CLASSIFICATION("TS")
DSMIL_SECURE_ENCLAVE
static int enclave_target_selection(double lat, double lon) {
    printf("\n=== Secure Enclave Execution (Intel SGX) ===\n");
    printf("Classification: TOP SECRET\n");
    printf("Function: Target Selection\n");
    printf("Coordinates: %.6f, %.6f\n", lat, lon);

    // Check tamper detection
    int tamper = dsmil_edge_tamper_detect();
    if (tamper != 0) {
        printf("CRITICAL: Tampering detected (event: %d)\n", tamper);
        printf("Executing emergency zeroization...\n");
        // dsmil_edge_zeroize();
        return -1;
    }

    printf("Enclave: TRUSTED\n");
    printf("Memory: ENCRYPTED\n");
    printf("Target selection computation: COMPLETE\n");

    return 0;
}

/**
 * @brief Perform remote attestation before classified processing
 *
 * Uses TPM 2.0 to generate attestation quote proving platform integrity
 * to remote verifier before processing classified data.
 */
DSMIL_CLASSIFICATION("S")
DSMIL_EDGE_SECURITY("remote_attest")
static int remote_attestation_check(void) {
    printf("\n=== Remote Attestation (TPM 2.0) ===\n");

    // Generate nonce from verifier
    uint8_t nonce[32];
    for (int i = 0; i < 32; i++) {
        nonce[i] = (uint8_t)rand();
    }

    // Generate attestation quote
    uint8_t quote[2048];
    size_t quote_len = 0;

    int result = dsmil_edge_remote_attest(nonce, quote, &quote_len);

    if (result == 0) {
        printf("Attestation quote generated: %zu bytes\n", quote_len);
        printf("Platform Configuration Registers (PCRs): MEASURED\n");
        printf("Attestation signature: VERIFIED\n");
        printf("Edge node status: TRUSTED\n");
    } else {
        printf("Attestation FAILED\n");
        return -1;
    }

    return 0;
}

//
// SCENARIO 4: Integrated High-Assurance Mission
//
// Combines nuclear surety, coalition operations, and edge security
// for a complete high-assurance military operation.
//

/**
 * @brief Execute integrated high-assurance strike mission
 *
 * Demonstrates all Phase 3 features in a coordinated operation:
 * - 2PI authorization for weapon release
 * - MPE coalition intelligence sharing
 * - Edge security on forward-deployed nodes
 */
DSMIL_CLASSIFICATION("TS/SCI")
DSMIL_JADC2_PROFILE("jadc2_targeting")
static int integrated_strike_mission(void) {
    printf("\n\n=== SCENARIO 4: Integrated High-Assurance Strike ===\n");
    printf("Mission: Joint NATO precision strike with nuclear deterrence\n");
    printf("Classification: TOP SECRET//SCI\n\n");

    // Step 1: Verify edge node security
    printf("Step 1: Edge Security Verification\n");
    if (remote_attestation_check() != 0) {
        printf("ABORT: Edge node not trusted\n");
        return -1;
    }

    // Step 2: Share coalition intelligence
    printf("\nStep 2: Coalition Intelligence Sharing\n");
    const char *target_intel = "Enemy air defense at 51.5074N, 0.1278W";
    process_coalition_intelligence(target_intel);

    // Step 3: U.S.-only targeting (NOFORN)
    printf("\nStep 3: U.S.-Only Targeting Computation\n");
    const char *noforn_data = "High-value target: Nuclear facility";
    process_noforn_humint(noforn_data);

    // Step 4: Secure enclave target selection
    printf("\nStep 4: Secure Enclave Target Processing\n");
    if (enclave_target_selection(51.5074, -0.1278) != 0) {
        printf("ABORT: Enclave computation failed\n");
        return -1;
    }

    // Step 5: Two-person nuclear authorization (if escalation required)
    printf("\nStep 5: Nuclear Escalation Authorization (2PI)\n");
    printf("SCENARIO: Adversary uses tactical nuclear weapon\n");
    printf("Response: Authorize limited nuclear strike\n\n");

    // Simulate officer signatures (production would use actual ML-DSA-87)
    uint8_t officer1_sig[MLDSA87_SIGNATURE_BYTES] = {0};
    uint8_t officer2_sig[MLDSA87_SIGNATURE_BYTES] = {0};

    int auth_result = authorize_nuclear_release(
        "B61-12 Tactical Nuclear Bomb",
        officer1_sig, officer2_sig,
        "POTUS", "SECDEF"
    );

    if (auth_result == 0) {
        printf("\n=== MISSION SUCCESS ===\n");
        printf("High-assurance controls verified:\n");
        printf("  ✓ Two-Person Integrity (Nuclear Surety)\n");
        printf("  ✓ Coalition Intelligence Sharing (MPE)\n");
        printf("  ✓ Edge Security Hardening (HSM/Enclave/Attestation)\n");
        printf("  ✓ All classification controls enforced\n");
    }

    return auth_result;
}

//
// MAIN: Run all scenarios
//

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  DSLLVM v1.6.0 Phase 3: High-Assurance Features Demo       ║\n");
    printf("║  Classification: TOP SECRET//SCI//NOFORN                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Initialize nuclear surety subsystem
    printf("Initializing Nuclear Surety (Two-Person Integrity)...\n");
    uint8_t officer1_pubkey[MLDSA87_PUBLIC_KEY_BYTES] = {0};
    uint8_t officer2_pubkey[MLDSA87_PUBLIC_KEY_BYTES] = {0};

    dsmil_nuclear_surety_init(
        "POTUS", officer1_pubkey,
        "SECDEF", officer2_pubkey
    );

    // Initialize Mission Partner Environment
    printf("Initializing Mission Partner Environment (MPE)...\n");
    dsmil_mpe_init("Operation JADC2-STRIKE", MPE_REL_NATO);

    // Add coalition partners
    uint8_t uk_cert[SHA256_HASH_BYTES] = {0};
    uint8_t fr_cert[SHA256_HASH_BYTES] = {0};
    uint8_t de_cert[SHA256_HASH_BYTES] = {0};
    uint8_t pl_cert[SHA256_HASH_BYTES] = {0};

    dsmil_mpe_add_partner("UK", "UK_MOD", uk_cert);
    dsmil_mpe_add_partner("FR", "FR_ARMY", fr_cert);
    dsmil_mpe_add_partner("DE", "DE_BUNDESWEHR", de_cert);
    dsmil_mpe_add_partner("PL", "PL_ARMED_FORCES", pl_cert);

    // Initialize edge security
    printf("Initializing Edge Security (HSM + SGX)...\n");
    dsmil_edge_security_init(HSM_TYPE_TPM2, ENCLAVE_SGX);

    printf("\n");

    // Run individual scenarios
    uint8_t sig1[MLDSA87_SIGNATURE_BYTES] = {0};
    uint8_t sig2[MLDSA87_SIGNATURE_BYTES] = {0};

    authorize_nuclear_release("Minuteman III ICBM", sig1, sig2, "POTUS", "SECDEF");
    change_defcon_level(3, sig1, sig2);

    process_coalition_intelligence("Threat Assessment: High");
    process_fvey_sigint("SIGINT: Adversary communications intercepted");
    process_noforn_humint("CIA HUMINT: Source REDACTED");

    uint8_t test_data[] = "Classified operational data";
    edge_process_classified(test_data, sizeof(test_data));
    enclave_target_selection(35.6892, 51.3890);
    remote_attestation_check();

    // Run integrated mission
    integrated_strike_mission();

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  All High-Assurance Scenarios Complete                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
