/**
 * @file dsmil_threat_signature.h
 * @brief DSLLVM Threat Signature Structures (v1.4)
 *
 * Threat signatures enable future AI-driven forensics by embedding
 * non-identifying fingerprints in binaries for correlation analysis.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_THREAT_SIGNATURE_H
#define DSMIL_THREAT_SIGNATURE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Threat signature version
 */
#define DSMIL_THREAT_SIGNATURE_VERSION 1

/**
 * Control-flow fingerprint
 */
typedef struct {
    char algorithm[32];       // "CFG-Merkle-Hash"
    uint8_t hash[48];        // SHA-384 hash
    uint32_t num_functions;   // Number of functions included
    char **function_names;    // Function names (NULL-terminated)
} dsmil_cfg_fingerprint_t;

/**
 * Crypto pattern information
 */
typedef struct {
    char algorithm[64];       // e.g., "ML-KEM-1024"
    char mode[32];           // e.g., "GCM"
    int constant_time_enforced;  // 1 if constant-time, 0 otherwise
} dsmil_crypto_pattern_t;

/**
 * Protocol schema information
 */
typedef struct {
    char protocol[64];        // e.g., "TLS-1.3"
    char **extensions;        // NULL-terminated array
    char **ciphersuites;      // NULL-terminated array
} dsmil_protocol_schema_t;

/**
 * Complete threat signature
 */
typedef struct {
    uint32_t version;         // DSMIL_THREAT_SIGNATURE_VERSION
    uint8_t binary_hash[48];  // SHA-384 of binary
    dsmil_cfg_fingerprint_t cfg;
    uint32_t num_crypto_patterns;
    dsmil_crypto_pattern_t *crypto_patterns;
    uint32_t num_protocol_schemas;
    dsmil_protocol_schema_t *protocol_schemas;
} dsmil_threat_signature_t;

/**
 * Extract threat signature from binary
 *
 * @param binary_path Path to binary file
 * @param signature Output threat signature
 * @return 0 on success, -1 on error
 */
int dsmil_extract_threat_signature(const char *binary_path,
                                   dsmil_threat_signature_t *signature);

/**
 * Compare two threat signatures
 *
 * @param sig1 First signature
 * @param sig2 Second signature
 * @return Similarity score (0.0 - 1.0)
 */
float dsmil_compare_threat_signatures(const dsmil_threat_signature_t *sig1,
                                      const dsmil_threat_signature_t *sig2);

/**
 * Free threat signature resources
 *
 * @param signature Threat signature to free
 */
void dsmil_free_threat_signature(dsmil_threat_signature_t *signature);

#ifdef __cplusplus
}
#endif

#endif // DSMIL_THREAT_SIGNATURE_H
