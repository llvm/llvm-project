/**
 * @file dsmil_provenance.h
 * @brief DSMIL Provenance Structures and API
 *
 * Defines structures and functions for CNSA 2.0 provenance records
 * embedded in DSLLVM-compiled binaries.
 *
 * Version: 1.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_PROVENANCE_H
#define DSMIL_PROVENANCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_PROV_CONSTANTS Constants
 * @{
 */

/** Maximum length of string fields */
#define DSMIL_PROV_MAX_STRING       256

/** Maximum number of build flags */
#define DSMIL_PROV_MAX_FLAGS        64

/** Maximum number of roles */
#define DSMIL_PROV_MAX_ROLES        16

/** Maximum number of section hashes */
#define DSMIL_PROV_MAX_SECTIONS     64

/** Maximum number of dependencies */
#define DSMIL_PROV_MAX_DEPS         32

/** Maximum certificate chain length */
#define DSMIL_PROV_MAX_CERT_CHAIN   5

/** SHA-384 hash size in bytes */
#define DSMIL_SHA384_SIZE           48

/** ML-DSA-87 signature size in bytes (FIPS 204) */
#define DSMIL_MLDSA87_SIG_SIZE      4627

/** ML-KEM-1024 ciphertext size in bytes (FIPS 203) */
#define DSMIL_MLKEM1024_CT_SIZE     1568

/** AES-256-GCM nonce size */
#define DSMIL_AES_GCM_NONCE_SIZE    12

/** AES-256-GCM tag size */
#define DSMIL_AES_GCM_TAG_SIZE      16

/** Provenance schema version */
#define DSMIL_PROV_SCHEMA_VERSION   "dsmil-provenance-v1"

/** @} */

/**
 * @defgroup DSMIL_PROV_ENUMS Enumerations
 * @{
 */

/** Hash algorithm identifiers */
typedef enum {
    DSMIL_HASH_SHA384 = 0,
    DSMIL_HASH_SHA512 = 1,
} dsmil_hash_alg_t;

/** Signature algorithm identifiers */
typedef enum {
    DSMIL_SIG_MLDSA87 = 0,  /**< ML-DSA-87 (FIPS 204) */
    DSMIL_SIG_MLDSA65 = 1,  /**< ML-DSA-65 (FIPS 204) */
} dsmil_sig_alg_t;

/** Key encapsulation algorithm identifiers */
typedef enum {
    DSMIL_KEM_MLKEM1024 = 0,  /**< ML-KEM-1024 (FIPS 203) */
    DSMIL_KEM_MLKEM768  = 1,  /**< ML-KEM-768 (FIPS 203) */
} dsmil_kem_alg_t;

/** Verification result codes */
typedef enum {
    DSMIL_VERIFY_OK                 = 0,   /**< Verification successful */
    DSMIL_VERIFY_NO_PROVENANCE      = 1,   /**< No provenance found */
    DSMIL_VERIFY_MALFORMED          = 2,   /**< Malformed provenance */
    DSMIL_VERIFY_UNSUPPORTED_ALG    = 3,   /**< Unsupported algorithm */
    DSMIL_VERIFY_UNKNOWN_SIGNER     = 4,   /**< Unknown signing key */
    DSMIL_VERIFY_CERT_INVALID       = 5,   /**< Invalid certificate chain */
    DSMIL_VERIFY_SIG_FAILED         = 6,   /**< Signature verification failed */
    DSMIL_VERIFY_HASH_MISMATCH      = 7,   /**< Binary hash mismatch */
    DSMIL_VERIFY_POLICY_VIOLATION   = 8,   /**< Policy violation */
    DSMIL_VERIFY_DECRYPT_FAILED     = 9,   /**< Decryption failed */
} dsmil_verify_result_t;

/** @} */

/**
 * @defgroup DSMIL_PROV_STRUCTS Data Structures
 * @{
 */

/** Compiler information */
typedef struct {
    char name[DSMIL_PROV_MAX_STRING];        /**< Compiler name (e.g., "dsmil-clang") */
    char version[DSMIL_PROV_MAX_STRING];     /**< Compiler version */
    char commit[DSMIL_PROV_MAX_STRING];      /**< Compiler build commit hash */
    char target[DSMIL_PROV_MAX_STRING];      /**< Target triple */
    uint8_t tsk_fingerprint[DSMIL_SHA384_SIZE]; /**< TSK fingerprint (SHA-384) */
} dsmil_compiler_info_t;

/** Source control information */
typedef struct {
    char vcs[32];                            /**< VCS type (e.g., "git") */
    char repo[DSMIL_PROV_MAX_STRING];        /**< Repository URL */
    char commit[DSMIL_PROV_MAX_STRING];      /**< Commit hash */
    char branch[DSMIL_PROV_MAX_STRING];      /**< Branch name */
    char tag[DSMIL_PROV_MAX_STRING];         /**< Tag (if any) */
    bool dirty;                              /**< Uncommitted changes present */
} dsmil_source_info_t;

/** Build information */
typedef struct {
    char timestamp[64];                      /**< ISO 8601 timestamp */
    char builder_id[DSMIL_PROV_MAX_STRING];  /**< Builder hostname/ID */
    uint8_t builder_cert[DSMIL_SHA384_SIZE]; /**< Builder cert fingerprint */
    char flags[DSMIL_PROV_MAX_FLAGS][DSMIL_PROV_MAX_STRING]; /**< Build flags */
    uint32_t num_flags;                      /**< Number of flags */
    bool reproducible;                       /**< Build is reproducible */
} dsmil_build_info_t;

/** DSMIL-specific metadata */
typedef struct {
    int32_t default_layer;                   /**< Default layer (0-8) */
    int32_t default_device;                  /**< Default device (0-103) */
    char roles[DSMIL_PROV_MAX_ROLES][64];    /**< Role names */
    uint32_t num_roles;                      /**< Number of roles */
    char sandbox_profile[128];               /**< Sandbox profile name */
    char stage[64];                          /**< MLOps stage */
    bool requires_npu;                       /**< Requires NPU */
    bool requires_gpu;                       /**< Requires GPU */
} dsmil_metadata_t;

/** Section hash entry */
typedef struct {
    char name[64];                           /**< Section name */
    uint8_t hash[DSMIL_SHA384_SIZE];         /**< SHA-384 hash */
} dsmil_section_hash_t;

/** Hash information */
typedef struct {
    dsmil_hash_alg_t algorithm;              /**< Hash algorithm */
    uint8_t binary[DSMIL_SHA384_SIZE];       /**< Binary hash (all PT_LOAD) */
    dsmil_section_hash_t sections[DSMIL_PROV_MAX_SECTIONS]; /**< Section hashes */
    uint32_t num_sections;                   /**< Number of sections */
} dsmil_hashes_t;

/** Dependency entry */
typedef struct {
    char name[DSMIL_PROV_MAX_STRING];        /**< Dependency name */
    uint8_t hash[DSMIL_SHA384_SIZE];         /**< SHA-384 hash */
    char version[64];                        /**< Version string */
} dsmil_dependency_t;

/** Certification information */
typedef struct {
    char fips_140_3[128];                    /**< FIPS 140-3 cert number */
    char common_criteria[128];               /**< Common Criteria EAL level */
    char supply_chain[128];                  /**< SLSA level */
} dsmil_certifications_t;

/** Complete provenance record */
typedef struct {
    char schema[64];                         /**< Schema version */
    char version[32];                        /**< Provenance format version */

    dsmil_compiler_info_t compiler;          /**< Compiler info */
    dsmil_source_info_t source;              /**< Source info */
    dsmil_build_info_t build;                /**< Build info */
    dsmil_metadata_t dsmil;                  /**< DSMIL metadata */
    dsmil_hashes_t hashes;                   /**< Hash values */

    dsmil_dependency_t dependencies[DSMIL_PROV_MAX_DEPS]; /**< Dependencies */
    uint32_t num_dependencies;               /**< Number of dependencies */

    dsmil_certifications_t certifications;   /**< Certifications */
} dsmil_provenance_t;

/** Signer information */
typedef struct {
    char key_id[DSMIL_PROV_MAX_STRING];      /**< Key ID */
    uint8_t fingerprint[DSMIL_SHA384_SIZE];  /**< Key fingerprint */
    uint8_t *cert_chain[DSMIL_PROV_MAX_CERT_CHAIN]; /**< Certificate chain */
    size_t cert_chain_lens[DSMIL_PROV_MAX_CERT_CHAIN]; /**< Cert lengths */
    uint32_t cert_chain_count;               /**< Number of certs */
} dsmil_signer_info_t;

/** RFC 3161 timestamp */
typedef struct {
    uint8_t *token;                          /**< RFC 3161 token */
    size_t token_len;                        /**< Token length */
    char authority[DSMIL_PROV_MAX_STRING];   /**< TSA URL */
} dsmil_timestamp_t;

/** Signature envelope (unencrypted) */
typedef struct {
    dsmil_provenance_t prov;                 /**< Provenance record */

    dsmil_hash_alg_t hash_alg;               /**< Hash algorithm */
    uint8_t prov_hash[DSMIL_SHA384_SIZE];    /**< Hash of canonical provenance */

    dsmil_sig_alg_t sig_alg;                 /**< Signature algorithm */
    uint8_t signature[DSMIL_MLDSA87_SIG_SIZE]; /**< Digital signature */
    size_t signature_len;                    /**< Actual signature length */

    dsmil_signer_info_t signer;              /**< Signer information */
    dsmil_timestamp_t timestamp;             /**< Optional timestamp */
} dsmil_signature_envelope_t;

/** Encrypted provenance envelope */
typedef struct {
    uint8_t *enc_prov;                       /**< Encrypted provenance (AEAD) */
    size_t enc_prov_len;                     /**< Ciphertext length */
    uint8_t tag[DSMIL_AES_GCM_TAG_SIZE];     /**< AEAD authentication tag */
    uint8_t nonce[DSMIL_AES_GCM_NONCE_SIZE]; /**< AEAD nonce */

    dsmil_kem_alg_t kem_alg;                 /**< KEM algorithm */
    uint8_t kem_ct[DSMIL_MLKEM1024_CT_SIZE]; /**< KEM ciphertext */
    size_t kem_ct_len;                       /**< Actual KEM ciphertext length */

    dsmil_hash_alg_t hash_alg;               /**< Hash algorithm */
    uint8_t prov_hash[DSMIL_SHA384_SIZE];    /**< Hash of encrypted envelope */

    dsmil_sig_alg_t sig_alg;                 /**< Signature algorithm */
    uint8_t signature[DSMIL_MLDSA87_SIG_SIZE]; /**< Digital signature */
    size_t signature_len;                    /**< Actual signature length */

    dsmil_signer_info_t signer;              /**< Signer information */
    dsmil_timestamp_t timestamp;             /**< Optional timestamp */
} dsmil_encrypted_envelope_t;

/** @} */

/**
 * @defgroup DSMIL_PROV_API API Functions
 * @{
 */

/**
 * @brief Extract provenance from ELF binary
 *
 * @param[in] binary_path Path to ELF binary
 * @param[out] envelope Output signature envelope (caller must free)
 * @return 0 on success, negative error code on failure
 */
int dsmil_extract_provenance(const char *binary_path,
                              dsmil_signature_envelope_t **envelope);

/**
 * @brief Verify provenance signature
 *
 * @param[in] envelope Signature envelope
 * @param[in] trust_store_path Path to trust store directory
 * @return Verification result code
 */
dsmil_verify_result_t dsmil_verify_provenance(
    const dsmil_signature_envelope_t *envelope,
    const char *trust_store_path);

/**
 * @brief Verify binary hash matches provenance
 *
 * @param[in] binary_path Path to ELF binary
 * @param[in] envelope Signature envelope
 * @return true if hash matches, false otherwise
 */
bool dsmil_verify_binary_hash(const char *binary_path,
                               const dsmil_signature_envelope_t *envelope);

/**
 * @brief Extract and decrypt provenance (ML-KEM-1024)
 *
 * @param[in] binary_path Path to ELF binary
 * @param[in] rdk_private_key RDK private key
 * @param[out] envelope Output signature envelope (caller must free)
 * @return 0 on success, negative error code on failure
 */
int dsmil_extract_encrypted_provenance(const char *binary_path,
                                        const void *rdk_private_key,
                                        dsmil_signature_envelope_t **envelope);

/**
 * @brief Free provenance envelope
 *
 * @param[in] envelope Envelope to free
 */
void dsmil_free_provenance(dsmil_signature_envelope_t *envelope);

/**
 * @brief Convert provenance to JSON
 *
 * @param[in] prov Provenance record
 * @param[out] json_out JSON string (caller must free)
 * @return 0 on success, negative error code on failure
 */
int dsmil_provenance_to_json(const dsmil_provenance_t *prov, char **json_out);

/**
 * @brief Convert verification result to string
 *
 * @param[in] result Verification result code
 * @return Human-readable string
 */
const char *dsmil_verify_result_str(dsmil_verify_result_t result);

/** @} */

/**
 * @defgroup DSMIL_PROV_BUILD Build-Time API
 * @{
 */

/**
 * @brief Build provenance record from metadata
 *
 * Called during link-time by dsmil-provenance-pass.
 *
 * @param[in] binary_path Path to output binary
 * @param[out] prov Output provenance record
 * @return 0 on success, negative error code on failure
 */
int dsmil_build_provenance(const char *binary_path, dsmil_provenance_t *prov);

/**
 * @brief Sign provenance with PSK
 *
 * @param[in] prov Provenance record
 * @param[in] psk_path Path to PSK private key
 * @param[out] envelope Output signature envelope
 * @return 0 on success, negative error code on failure
 */
int dsmil_sign_provenance(const dsmil_provenance_t *prov,
                           const char *psk_path,
                           dsmil_signature_envelope_t *envelope);

/**
 * @brief Encrypt and sign provenance with PSK + RDK
 *
 * @param[in] prov Provenance record
 * @param[in] psk_path Path to PSK private key
 * @param[in] rdk_pub_path Path to RDK public key
 * @param[out] enc_envelope Output encrypted envelope
 * @return 0 on success, negative error code on failure
 */
int dsmil_encrypt_sign_provenance(const dsmil_provenance_t *prov,
                                   const char *psk_path,
                                   const char *rdk_pub_path,
                                   dsmil_encrypted_envelope_t *enc_envelope);

/**
 * @brief Embed provenance envelope in ELF binary
 *
 * @param[in] binary_path Path to ELF binary (modified in-place)
 * @param[in] envelope Signature envelope
 * @return 0 on success, negative error code on failure
 */
int dsmil_embed_provenance(const char *binary_path,
                            const dsmil_signature_envelope_t *envelope);

/**
 * @brief Embed encrypted provenance envelope in ELF binary
 *
 * @param[in] binary_path Path to ELF binary (modified in-place)
 * @param[in] enc_envelope Encrypted envelope
 * @return 0 on success, negative error code on failure
 */
int dsmil_embed_encrypted_provenance(const char *binary_path,
                                      const dsmil_encrypted_envelope_t *enc_envelope);

/** @} */

/**
 * @defgroup DSMIL_PROV_UTIL Utility Functions
 * @{
 */

/**
 * @brief Get current build timestamp (ISO 8601)
 *
 * @param[out] timestamp Output buffer (min 64 bytes)
 * @return 0 on success, negative error code on failure
 */
int dsmil_get_build_timestamp(char *timestamp);

/**
 * @brief Get Git repository information
 *
 * @param[in] repo_path Path to Git repository
 * @param[out] source_info Output source info
 * @return 0 on success, negative error code on failure
 */
int dsmil_get_git_info(const char *repo_path, dsmil_source_info_t *source_info);

/**
 * @brief Compute SHA-384 hash of file
 *
 * @param[in] file_path Path to file
 * @param[out] hash Output hash (48 bytes)
 * @return 0 on success, negative error code on failure
 */
int dsmil_hash_file_sha384(const char *file_path, uint8_t hash[DSMIL_SHA384_SIZE]);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_PROVENANCE_H */
