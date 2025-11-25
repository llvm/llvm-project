#include "../include/tpm2_compat_accelerated.h"

// TSS2 headers commented out - not available in this build environment
// #include <tss2/tss2_esys.h>
// #include <tss2/tss2_rc.h>
// #include <tss2/tss2_mu.h>

// Stub definitions for TSS2 types when library not available
#ifndef TSS2_RC_SUCCESS
typedef uint32_t TSS2_RC;
#define TSS2_RC_SUCCESS 0

typedef void* ESYS_CONTEXT;
typedef uint32_t ESYS_TR;
#define ESYS_TR_NONE 0
#define ESYS_TR_RH_OWNER 0x40000001
#define ESYS_TR_RH_ENDORSEMENT 0x4000000B
#define ESYS_TR_RH_PLATFORM 0x4000000C
#define ESYS_TR_PASSWORD 0x40000009
#define ESYS_TR_PCR0 0x00

// Define simple types first
typedef uint16_t TPM2_ALG_ID;
typedef uint16_t TPM2_ECC_CURVE;
typedef uint32_t TPMA_OBJECT;
typedef uint16_t TPM2_ST;

// Define basic buffer types
typedef struct { uint16_t size; uint8_t buffer[256]; } TPM2B_PRIVATE;
typedef struct { uint16_t size; uint8_t buffer[64]; } TPM2B_DIGEST;
typedef struct { uint16_t size; uint8_t buffer[64]; } TPM2B_DATA;
typedef struct { uint16_t size; uint8_t buffer[64]; } TPM2B_AUTH;
typedef struct { uint16_t size; uint8_t buffer[256]; } TPM2B_ATTEST;
typedef struct { uint16_t size; uint8_t buffer[32]; } TPM2B_ECC_PARAMETER;

// Define TPMT_PUBLIC and TPM2B_PUBLIC
typedef struct {
    TPM2_ALG_ID type;
    TPM2_ALG_ID nameAlg;
    TPMA_OBJECT objectAttributes;
    TPM2B_DIGEST authPolicy;
    union {
        struct {
            struct { TPM2_ALG_ID algorithm; uint16_t keyBits; uint16_t mode; } symmetric;
            struct { TPM2_ALG_ID scheme; } scheme;
            uint16_t keyBits;
            uint32_t exponent;
        } rsaDetail;
        struct {
            struct { TPM2_ALG_ID algorithm; } symmetric;
            struct { TPM2_ALG_ID scheme; struct { TPM2_ALG_ID hashAlg; } details; } scheme;
            TPM2_ECC_CURVE curveID;
            struct { TPM2_ALG_ID scheme; } kdf;
        } eccDetail;
    } parameters;
    union {
        struct { uint16_t size; uint8_t buffer[256]; } rsa;
        struct { TPM2B_ECC_PARAMETER x; TPM2B_ECC_PARAMETER y; } ecc;
    } unique;
} TPMT_PUBLIC;
typedef struct { uint16_t size; TPMT_PUBLIC publicArea; } TPM2B_PUBLIC;

#define TPM2_ALG_RSA 0x0001
#define TPM2_ALG_AES 0x0006
#define TPM2_ALG_SHA256 0x000B
#define TPM2_ALG_SHA384 0x000C
#define TPM2_ALG_SHA512 0x000D
#define TPM2_ALG_NULL 0x0010
#define TPM2_ALG_CFB 0x0043
#define TPM2_ALG_ECC 0x0023
#define TPM2_ALG_ECDSA 0x0018
#define TPM2_ECC_NIST_P256 0x0003
#define TPM2_ECC_NIST_P384 0x0004
#define TPM2_ECC_NIST_P521 0x0005
#define TPMA_OBJECT_RESTRICTED 0x00010000
#define TPMA_OBJECT_DECRYPT 0x00020000
#define TPMA_OBJECT_FIXEDTPM 0x00000002
#define TPMA_OBJECT_FIXEDPARENT 0x00000010
#define TPMA_OBJECT_SENSITIVEDATAORIGIN 0x00000020
#define TPMA_OBJECT_USERWITHAUTH 0x00000040
#define TPMA_OBJECT_SIGN_ENCRYPT 0x00040000
#define TPM2_ST_ATTEST_QUOTE 0x8018

typedef struct { uint32_t count; void *pcrSelections; } TPML_PCR_SELECTION;
typedef struct { TPML_PCR_SELECTION pcrSelect; TPM2B_DIGEST pcrDigest; } TPMS_QUOTE_INFO;
typedef struct { TPM2_ST type; TPM2B_DATA extraData; union { TPMS_QUOTE_INFO quote; } attested; } TPMS_ATTEST;
typedef struct { TPM2_ALG_ID hash; uint8_t sizeofSelect; uint8_t pcrSelect[3]; } TPMS_PCR_SELECTION;
typedef struct { TPM2_ALG_ID sigAlg; union { struct { TPM2_ALG_ID hash; TPM2B_ECC_PARAMETER signatureR; TPM2B_ECC_PARAMETER signatureS; } ecdsa; } signature; } TPMT_SIGNATURE;
typedef struct { TPM2_ALG_ID scheme; } TPMT_SIG_SCHEME;
typedef struct { uint32_t count; struct { TPM2_ALG_ID hashAlg; union { uint8_t sha256[32]; } digest; } digests[8]; } TPML_DIGEST_VALUES;
typedef struct { size_t size; struct { TPM2B_AUTH userAuth; TPM2B_DATA data; } sensitive; } TPM2B_SENSITIVE_CREATE;

// Stub functions
static inline TSS2_RC Esys_Initialize(ESYS_CONTEXT **ctx, void *a, void *b) { (void)a; (void)b; *ctx = NULL; return 1; /* failure */ }
static inline void Esys_Finalize(ESYS_CONTEXT **ctx) { (void)ctx; }
static inline TSS2_RC Esys_TR_SetAuth(ESYS_CONTEXT *ctx, ESYS_TR tr, const TPM2B_AUTH *auth) { (void)ctx; (void)tr; (void)auth; return TSS2_RC_SUCCESS; }
static inline TSS2_RC Esys_CreatePrimary(ESYS_CONTEXT *ctx, ESYS_TR a, ESYS_TR b, ESYS_TR c, ESYS_TR d, const TPM2B_SENSITIVE_CREATE *e, const TPM2B_PUBLIC *f, const TPM2B_DATA *g, const TPML_PCR_SELECTION *h, ESYS_TR *i, void *j, void *k, void *l, void *m) { (void)ctx; (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; (void)j; (void)k; (void)l; (void)m; return 1; }
static inline TSS2_RC Esys_Create(ESYS_CONTEXT *ctx, ESYS_TR a, ESYS_TR b, ESYS_TR c, ESYS_TR d, const TPM2B_SENSITIVE_CREATE *e, const TPM2B_PUBLIC *f, const TPM2B_DATA *g, const TPML_PCR_SELECTION *h, TPM2B_PRIVATE **i, TPM2B_PUBLIC **j, void *k, void *l, void *m) { (void)ctx; (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; (void)j; (void)k; (void)l; (void)m; return 1; }
static inline TSS2_RC Esys_Load(ESYS_CONTEXT *ctx, ESYS_TR a, ESYS_TR b, ESYS_TR c, ESYS_TR d, const TPM2B_PRIVATE *e, const TPM2B_PUBLIC *f, ESYS_TR *g) { (void)ctx; (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return 1; }
static inline TSS2_RC Esys_Quote(ESYS_CONTEXT *ctx, ESYS_TR a, ESYS_TR b, ESYS_TR c, ESYS_TR d, const TPM2B_DATA *e, const TPMT_SIG_SCHEME *f, const TPML_PCR_SELECTION *g, TPM2B_ATTEST **h, TPMT_SIGNATURE **i) { (void)ctx; (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; return 1; }
static inline TSS2_RC Esys_PCR_Extend(ESYS_CONTEXT *ctx, ESYS_TR a, ESYS_TR b, ESYS_TR c, ESYS_TR d, const TPML_DIGEST_VALUES *e) { (void)ctx; (void)a; (void)b; (void)c; (void)d; (void)e; return 1; }
static inline TSS2_RC Esys_FlushContext(ESYS_CONTEXT *ctx, ESYS_TR tr) { (void)ctx; (void)tr; return TSS2_RC_SUCCESS; }
static inline void Esys_Free(void *ptr) { (void)ptr; }
static inline TSS2_RC Tss2_MU_TPMS_ATTEST_Unmarshal(const uint8_t *buf, size_t len, size_t *off, TPMS_ATTEST *out) { (void)buf; (void)len; (void)off; (void)out; return 1; }
#endif

#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <openssl/crypto.h>
#include <openssl/x509.h>

#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define TPM2_AGENT_PCR_INDEX 23u
#define TPM2_DIGEST_SIZE     32u

struct tpm2_crypto_context_t {
    tpm2_crypto_algorithm_t algorithm;
    uint8_t key_material[64];
    size_t key_size;
};

static ESYS_CONTEXT *g_esys = NULL;
static ESYS_TR g_primary = ESYS_TR_NONE;
static ESYS_TR g_ak = ESYS_TR_NONE;
static TPM2B_PUBLIC g_ak_public = {0};
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_initialized = false;
static char g_active_agent[TPM2_AGENT_MAX_ID_LENGTH] = {0};
static uint8_t g_active_nonce[TPM2_DIGEST_SIZE] = {0};
static size_t g_active_nonce_len = 0;

static tpm2_rc_t map_tss2_rc(TSS2_RC rc);
static void reset_state_locked(void);
static tpm2_rc_t ensure_initialized_locked(void);
static tpm2_rc_t create_primary_locked(void);
static tpm2_rc_t create_attestation_key_locked(void);
static const EVP_MD *map_hash_alg_to_evp(tpm2_crypto_algorithm_t alg);
static const EVP_MD *map_tpm_hash_to_evp(uint16_t alg);
static const EVP_CIPHER *map_cipher_to_evp(tpm2_crypto_algorithm_t alg, size_t key_len);
static tpm2_rc_t compute_task_nonce(const char *agent_id, const uint8_t *payload, size_t payload_len, uint8_t *digest_out, size_t *digest_len);
static tpm2_rc_t extend_task_digest_locked(const uint8_t *digest, size_t digest_len);
static EVP_PKEY *evp_from_tpm_public(const TPM2B_PUBLIC *public_area);
static int map_tpm_ecc_curve_to_nid(TPM2_ECC_CURVE curve);
static tpm2_rc_t export_attestation_public(uint8_t *buffer, size_t *len);
static tpm2_rc_t encode_signature_der(const TPMT_SIGNATURE *signature, uint8_t *buffer, size_t *len);
static tpm2_rc_t verify_signature_der(uint16_t hash_alg, const uint8_t *public_key, size_t public_key_len, const uint8_t *signature, size_t signature_len, const uint8_t *data, size_t data_len);
static tpm2_rc_t parse_attestation_blob(const uint8_t *blob, size_t blob_len, TPMS_ATTEST *decoded);
static void clear_agent_state_locked(void);

tpm2_rc_t tpm2_crypto_init(tpm2_acceleration_flags_t acceleration, tpm2_security_level_t level) {
    (void)acceleration;
    (void)level;

    pthread_mutex_lock(&g_lock);
    tpm2_rc_t rc = ensure_initialized_locked();
    pthread_mutex_unlock(&g_lock);
    return rc;
}

tpm2_rc_t tpm2_crypto_context_create(tpm2_crypto_algorithm_t algorithm, const uint8_t *key_material, size_t key_size, tpm2_crypto_context_handle_t *context_out) {
    if (!context_out || !key_material || key_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_CIPHER *cipher = map_cipher_to_evp(algorithm, key_size);
    if (!cipher) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    struct tpm2_crypto_context_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return TPM2_RC_FAILURE;
    }

    if (key_size > sizeof(ctx->key_material)) {
        free(ctx);
        return TPM2_RC_BAD_PARAMETER;
    }

    memcpy(ctx->key_material, key_material, key_size);
    ctx->key_size = key_size;
    ctx->algorithm = algorithm;
    *context_out = ctx;
    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_crypto_hash_accelerated(tpm2_crypto_algorithm_t hash_alg, const uint8_t *data, size_t data_size, uint8_t *hash_out, size_t *hash_size_inout) {
    if (!data || !hash_out || !hash_size_inout) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_MD *md = map_hash_alg_to_evp(hash_alg);
    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    unsigned int digest_len = 0;
    if (*hash_size_inout < (size_t)EVP_MD_size(md)) {
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    if (EVP_Digest(data, data_size, hash_out, &digest_len, md, NULL) != 1) {
        return TPM2_RC_FAILURE;
    }

    *hash_size_inout = digest_len;
    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_crypto_encrypt_accelerated(tpm2_crypto_context_handle_t context_handle, const uint8_t *plaintext, size_t plaintext_size, const uint8_t *iv, size_t iv_size, uint8_t *ciphertext_out, size_t *ciphertext_size_inout) {
    if (!context_handle || !plaintext || !iv || !ciphertext_out || !ciphertext_size_inout) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const struct tpm2_crypto_context_t *ctx = context_handle;
    const EVP_CIPHER *cipher = map_cipher_to_evp(ctx->algorithm, ctx->key_size);
    if (!cipher) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    EVP_CIPHER_CTX *cipher_ctx = EVP_CIPHER_CTX_new();
    if (!cipher_ctx) {
        return TPM2_RC_FAILURE;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    int out_len = 0, total_len = 0;
    int block_len = EVP_CIPHER_block_size(cipher);

    if ((size_t)EVP_CIPHER_iv_length(cipher) != iv_size) {
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    if (*ciphertext_size_inout < plaintext_size + (size_t)block_len) {
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }

    if (EVP_EncryptInit_ex(cipher_ctx, cipher, NULL, ctx->key_material, iv) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    if (EVP_EncryptUpdate(cipher_ctx, ciphertext_out, &out_len, plaintext, (int)plaintext_size) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    total_len = out_len;
    if (EVP_EncryptFinal_ex(cipher_ctx, ciphertext_out + total_len, &out_len) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    total_len += out_len;
    *ciphertext_size_inout = (size_t)total_len;

cleanup:
    EVP_CIPHER_CTX_free(cipher_ctx);
    return rc;
}

tpm2_rc_t tpm2_crypto_decrypt_accelerated(tpm2_crypto_context_handle_t context_handle, const uint8_t *ciphertext, size_t ciphertext_size, const uint8_t *iv, size_t iv_size, uint8_t *plaintext_out, size_t *plaintext_size_inout) {
    if (!context_handle || !ciphertext || !iv || !plaintext_out || !plaintext_size_inout) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const struct tpm2_crypto_context_t *ctx = context_handle;
    const EVP_CIPHER *cipher = map_cipher_to_evp(ctx->algorithm, ctx->key_size);
    if (!cipher) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    EVP_CIPHER_CTX *cipher_ctx = EVP_CIPHER_CTX_new();
    if (!cipher_ctx) {
        return TPM2_RC_FAILURE;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    int out_len = 0, total_len = 0;

    if ((size_t)EVP_CIPHER_iv_length(cipher) != iv_size) {
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    if (*plaintext_size_inout < ciphertext_size) {
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }

    if (EVP_DecryptInit_ex(cipher_ctx, cipher, NULL, ctx->key_material, iv) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    if (EVP_DecryptUpdate(cipher_ctx, plaintext_out, &out_len, ciphertext, (int)ciphertext_size) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    total_len = out_len;
    if (EVP_DecryptFinal_ex(cipher_ctx, plaintext_out + total_len, &out_len) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    total_len += out_len;
    *plaintext_size_inout = (size_t)total_len;

cleanup:
    EVP_CIPHER_CTX_free(cipher_ctx);
    return rc;
}

tpm2_rc_t tpm2_crypto_sign_accelerated(tpm2_crypto_context_handle_t context, const uint8_t *digest, size_t digest_len, uint8_t *signature_out, size_t *signature_len_inout) {
    (void)context;
    (void)digest;
    (void)digest_len;
    (void)signature_out;
    (void)signature_len_inout;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_verify_accelerated(tpm2_crypto_context_handle_t context, const uint8_t *digest, size_t digest_len, const uint8_t *signature, size_t signature_len, bool *valid_out) {
    (void)context;
    (void)digest;
    (void)digest_len;
    (void)signature;
    (void)signature_len;
    if (valid_out) {
        *valid_out = false;
    }
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_context_destroy(tpm2_crypto_context_handle_t context_handle) {
    if (!context_handle) {
        return TPM2_RC_BAD_PARAMETER;
    }
    struct tpm2_crypto_context_t *ctx = context_handle;
    OPENSSL_cleanse(ctx->key_material, sizeof(ctx->key_material));
    free(ctx);
    return TPM2_RC_SUCCESS;
}

void tpm2_crypto_cleanup(void) {
    pthread_mutex_lock(&g_lock);
    reset_state_locked();
    pthread_mutex_unlock(&g_lock);
}

tpm2_rc_t tpm2_agent_task_begin(const char *agent_id, const uint8_t *task_descriptor, size_t descriptor_len) {
    if (!agent_id || agent_id[0] == '\0' || strlen(agent_id) >= TPM2_AGENT_MAX_ID_LENGTH) {
        return TPM2_RC_BAD_PARAMETER;
    }

    uint8_t nonce[TPM2_DIGEST_SIZE] = {0};
    size_t nonce_len = sizeof(nonce);

    pthread_mutex_lock(&g_lock);
    tpm2_rc_t rc = ensure_initialized_locked();
    if (rc != TPM2_RC_SUCCESS) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    rc = compute_task_nonce(agent_id, task_descriptor, descriptor_len, nonce, &nonce_len);
    if (rc != TPM2_RC_SUCCESS) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    rc = extend_task_digest_locked(nonce, nonce_len);
    if (rc == TPM2_RC_SUCCESS) {
        strncpy(g_active_agent, agent_id, sizeof(g_active_agent) - 1);
        g_active_agent[sizeof(g_active_agent) - 1] = '\0';
        memcpy(g_active_nonce, nonce, nonce_len);
        g_active_nonce_len = nonce_len;
    }

    pthread_mutex_unlock(&g_lock);
    return rc;
}

tpm2_rc_t tpm2_agent_task_complete(const char *agent_id, const uint8_t *result_digest, size_t result_digest_len, tpm2_agent_attestation_t *attestation_out) {
    if (!agent_id || !attestation_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    uint8_t final_nonce[TPM2_DIGEST_SIZE] = {0};
    size_t final_nonce_len = sizeof(final_nonce);

    pthread_mutex_lock(&g_lock);
    tpm2_rc_t rc = ensure_initialized_locked();
    if (rc != TPM2_RC_SUCCESS) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    if (strncmp(g_active_agent, agent_id, TPM2_AGENT_MAX_ID_LENGTH) != 0 || g_active_nonce_len == 0) {
        pthread_mutex_unlock(&g_lock);
        return TPM2_RC_INVALID_STATE;
    }

    rc = compute_task_nonce(agent_id, result_digest, result_digest_len, final_nonce, &final_nonce_len);
    if (rc != TPM2_RC_SUCCESS) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    rc = extend_task_digest_locked(final_nonce, final_nonce_len);
    if (rc != TPM2_RC_SUCCESS) {
        pthread_mutex_unlock(&g_lock);
        return rc;
    }

    TPM2B_DATA qualifying_data = {.size = (uint16_t)final_nonce_len};
    memcpy(qualifying_data.buffer, final_nonce, final_nonce_len);

    TPML_PCR_SELECTION pcr_selection = {.count = 1};
    pcr_selection.pcrSelections[0].hash = TPM2_ALG_SHA256;
    pcr_selection.pcrSelections[0].sizeofSelect = 3;
    memset(pcr_selection.pcrSelections[0].pcrSelect, 0, sizeof(pcr_selection.pcrSelections[0].pcrSelect));
    pcr_selection.pcrSelections[0].pcrSelect[TPM2_AGENT_PCR_INDEX / 8] |= (1u << (TPM2_AGENT_PCR_INDEX % 8));

    TPMT_SIG_SCHEME sig_scheme = {.scheme = TPM2_ALG_NULL};
    TPM2B_ATTEST *attest_blob = NULL;
    TPMT_SIGNATURE *signature = NULL;

    TSS2_RC tss_rc = Esys_Quote(
        g_esys,
        g_ak,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        &qualifying_data,
        &sig_scheme,
        &pcr_selection,
        &attest_blob,
        &signature
    );

    if (tss_rc != TSS2_RC_SUCCESS) {
        rc = map_tss2_rc(tss_rc);
        goto cleanup;
    }

    TPMS_ATTEST decoded = {0};
    rc = parse_attestation_blob(attest_blob->attestationData, attest_blob->size, &decoded);
    if (rc != TPM2_RC_SUCCESS) {
        goto cleanup;
    }

    if (decoded.type != TPM2_ST_ATTEST_QUOTE) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    if (decoded.extraData.size != final_nonce_len || memcmp(decoded.extraData.buffer, final_nonce, final_nonce_len) != 0) {
        rc = TPM2_RC_SECURITY_VIOLATION;
        goto cleanup;
    }

    bool pcr_found = false;
    for (uint32_t i = 0; i < decoded.attested.quote.pcrSelect.count; ++i) {
        const TPMS_PCR_SELECTION *sel = &decoded.attested.quote.pcrSelect.pcrSelections[i];
        if (sel->hash != TPM2_ALG_SHA256) {
            continue;
        }
        uint32_t byte_idx = TPM2_AGENT_PCR_INDEX / 8;
        uint8_t mask = (uint8_t)(1u << (TPM2_AGENT_PCR_INDEX % 8));
        if (byte_idx < sel->sizeofSelect && (sel->pcrSelect[byte_idx] & mask)) {
            pcr_found = true;
            break;
        }
    }

    if (!pcr_found) {
        rc = TPM2_RC_SECURITY_VIOLATION;
        goto cleanup;
    }

    memset(attestation_out, 0, sizeof(*attestation_out));
    strncpy(attestation_out->agent_id, agent_id, sizeof(attestation_out->agent_id) - 1);
    attestation_out->pcr_index = TPM2_AGENT_PCR_INDEX;
    attestation_out->hash_algorithm = (signature->sigAlg == TPM2_ALG_ECDSA)
        ? signature->signature.ecdsa.hash
        : TPM2_ALG_SHA256;

    attestation_out->task_nonce_len = final_nonce_len;
    memcpy(attestation_out->task_nonce, final_nonce, final_nonce_len);

    attestation_out->pcr_digest_len = decoded.attested.quote.pcrDigest.size;
    memcpy(attestation_out->pcr_digest, decoded.attested.quote.pcrDigest.buffer, decoded.attested.quote.pcrDigest.size);

    attestation_out->attestation_blob_len = attest_blob->size;
    if (attestation_out->attestation_blob_len > TPM2_AGENT_MAX_ATTEST_BLOB) {
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }
    memcpy(attestation_out->attestation_blob, attest_blob->attestationData, attest_blob->size);

    {
        size_t sig_len = TPM2_AGENT_MAX_SIGNATURE;
        rc = encode_signature_der(signature, attestation_out->signature, &sig_len);
        if (rc != TPM2_RC_SUCCESS) {
            goto cleanup;
        }
        attestation_out->signature_len = sig_len;
    }

    {
        size_t pub_len = TPM2_AGENT_MAX_PUBLIC_KEY;
        rc = export_attestation_public(attestation_out->public_key, &pub_len);
        if (rc != TPM2_RC_SUCCESS) {
            goto cleanup;
        }
        attestation_out->public_key_len = pub_len;
    }

    clear_agent_state_locked();
    rc = TPM2_RC_SUCCESS;

cleanup:
    if (attest_blob) {
        Esys_Free(attest_blob);
    }
    if (signature) {
        Esys_Free(signature);
    }
    pthread_mutex_unlock(&g_lock);
    return rc;
}

tpm2_rc_t tpm2_agent_task_verify(const char *agent_id, const uint8_t *expected_result_digest, size_t result_digest_len, const tpm2_agent_attestation_t *attestation) {
    if (!agent_id || !attestation || attestation->attestation_blob_len == 0 || attestation->signature_len == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    if (strncmp(attestation->agent_id, agent_id, TPM2_AGENT_MAX_ID_LENGTH) != 0) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    uint8_t expected_nonce[TPM2_DIGEST_SIZE] = {0};
    size_t expected_nonce_len = sizeof(expected_nonce);
    tpm2_rc_t rc = compute_task_nonce(agent_id, expected_result_digest, result_digest_len, expected_nonce, &expected_nonce_len);
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    if (attestation->task_nonce_len != expected_nonce_len || memcmp(attestation->task_nonce, expected_nonce, expected_nonce_len) != 0) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    TPMS_ATTEST decoded = {0};
    rc = parse_attestation_blob(attestation->attestation_blob, attestation->attestation_blob_len, &decoded);
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    if (decoded.type != TPM2_ST_ATTEST_QUOTE) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    if (decoded.extraData.size != expected_nonce_len || memcmp(decoded.extraData.buffer, expected_nonce, expected_nonce_len) != 0) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    if (decoded.attested.quote.pcrDigest.size != attestation->pcr_digest_len ||
        memcmp(decoded.attested.quote.pcrDigest.buffer, attestation->pcr_digest, attestation->pcr_digest_len) != 0) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    bool pcr_found = false;
    for (uint32_t i = 0; i < decoded.attested.quote.pcrSelect.count; ++i) {
        const TPMS_PCR_SELECTION *sel = &decoded.attested.quote.pcrSelect.pcrSelections[i];
        if (sel->hash != TPM2_ALG_SHA256) {
            continue;
        }
        uint32_t byte_idx = attestation->pcr_index / 8;
        uint8_t mask = (uint8_t)(1u << (attestation->pcr_index % 8));
        if (byte_idx < sel->sizeofSelect && (sel->pcrSelect[byte_idx] & mask)) {
            pcr_found = true;
            break;
        }
    }

    if (!pcr_found) {
        return TPM2_RC_SECURITY_VIOLATION;
    }

    rc = verify_signature_der(
        attestation->hash_algorithm,
        attestation->public_key,
        attestation->public_key_len,
        attestation->signature,
        attestation->signature_len,
        attestation->attestation_blob,
        attestation->attestation_blob_len
    );

    return rc;
}

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

static tpm2_rc_t map_tss2_rc(TSS2_RC rc) {
    return (rc == TSS2_RC_SUCCESS) ? TPM2_RC_SUCCESS : TPM2_RC_FAILURE;
}

static void reset_state_locked(void) {
    if (g_esys) {
        if (g_ak != ESYS_TR_NONE) {
            Esys_FlushContext(g_esys, g_ak);
        }
        if (g_primary != ESYS_TR_NONE) {
            Esys_FlushContext(g_esys, g_primary);
        }
        Esys_Finalize(&g_esys);
        g_esys = NULL;
    }

    g_primary = ESYS_TR_NONE;
    g_ak = ESYS_TR_NONE;
    memset(&g_ak_public, 0, sizeof(g_ak_public));
    g_initialized = false;
    clear_agent_state_locked();
}

static tpm2_rc_t ensure_initialized_locked(void) {
    if (g_initialized) {
        return TPM2_RC_SUCCESS;
    }

    TSS2_RC rc = Esys_Initialize(&g_esys, NULL, NULL);
    if (rc != TSS2_RC_SUCCESS) {
        reset_state_locked();
        return map_tss2_rc(rc);
    }

    TPM2B_AUTH empty_auth = {.size = 0};
    Esys_TR_SetAuth(g_esys, ESYS_TR_RH_OWNER, &empty_auth);
    Esys_TR_SetAuth(g_esys, ESYS_TR_RH_ENDORSEMENT, &empty_auth);
    Esys_TR_SetAuth(g_esys, ESYS_TR_RH_PLATFORM, &empty_auth);

    tpm2_rc_t primer = create_primary_locked();
    if (primer != TPM2_RC_SUCCESS) {
        reset_state_locked();
        return primer;
    }

    tpm2_rc_t ak_rc = create_attestation_key_locked();
    if (ak_rc != TPM2_RC_SUCCESS) {
        reset_state_locked();
        return ak_rc;
    }

    g_initialized = true;
    clear_agent_state_locked();
    return TPM2_RC_SUCCESS;
}

static tpm2_rc_t create_primary_locked(void) {
    TPM2B_SENSITIVE_CREATE in_sensitive = {
        .size = sizeof(in_sensitive.sensitive),
        .sensitive = {
            .userAuth = {.size = 0},
            .data = {.size = 0}
        }
    };

    TPM2B_PUBLIC in_public = {
        .size = 0,
        .publicArea = {
            .type = TPM2_ALG_RSA,
            .nameAlg = TPM2_ALG_SHA256,
            .objectAttributes = TPMA_OBJECT_RESTRICTED |
                                TPMA_OBJECT_DECRYPT |
                                TPMA_OBJECT_FIXEDTPM |
                                TPMA_OBJECT_FIXEDPARENT |
                                TPMA_OBJECT_SENSITIVEDATAORIGIN |
                                TPMA_OBJECT_USERWITHAUTH,
            .parameters.rsaDetail = {
                .symmetric = {
                    .algorithm = TPM2_ALG_AES,
                    .keyBits.aes = 128,
                    .mode.aes = TPM2_ALG_CFB
                },
                .scheme = {
                    .scheme = TPM2_ALG_NULL
                },
                .keyBits = 2048,
                .exponent = 0
            },
            .unique.rsa = {.size = 0}
        }
    };

    TPM2B_DATA outside_info = {.size = 0};
    TPML_PCR_SELECTION creation_pcr = {.count = 0};

    TSS2_RC rc = Esys_CreatePrimary(
        g_esys,
        ESYS_TR_RH_OWNER,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        &in_sensitive,
        &in_public,
        &outside_info,
        &creation_pcr,
        &g_primary,
        NULL,
        NULL,
        NULL,
        NULL
    );

    if (rc != TSS2_RC_SUCCESS) {
        return map_tss2_rc(rc);
    }

    TPM2B_AUTH empty_auth = {.size = 0};
    Esys_TR_SetAuth(g_esys, g_primary, &empty_auth);
    return TPM2_RC_SUCCESS;
}

static tpm2_rc_t create_attestation_key_locked(void) {
    TPM2B_SENSITIVE_CREATE in_sensitive = {
        .size = sizeof(in_sensitive.sensitive),
        .sensitive = {
            .userAuth = {.size = 0},
            .data = {.size = 0}
        }
    };

    TPM2B_PUBLIC in_public = {
        .size = 0,
        .publicArea = {
            .type = TPM2_ALG_ECC,
            .nameAlg = TPM2_ALG_SHA256,
            .objectAttributes = TPMA_OBJECT_SIGN_ENCRYPT |
                                TPMA_OBJECT_FIXEDTPM |
                                TPMA_OBJECT_FIXEDPARENT |
                                TPMA_OBJECT_SENSITIVEDATAORIGIN |
                                TPMA_OBJECT_USERWITHAUTH,
            .parameters.eccDetail = {
                .symmetric = {.algorithm = TPM2_ALG_NULL},
                .scheme = {
                    .scheme = TPM2_ALG_ECDSA,
                    .details.ecdsa.hashAlg = TPM2_ALG_SHA256
                },
                .curveID = TPM2_ECC_NIST_P256,
                .kdf = {.scheme = TPM2_ALG_NULL}
            },
            .unique.ecc = {
                .x = {.size = 0},
                .y = {.size = 0}
            }
        }
    };

    TPM2B_DATA outside_info = {.size = 0};
    TPML_PCR_SELECTION creation_pcr = {.count = 0};

    TPM2B_PRIVATE *out_private = NULL;
    TPM2B_PUBLIC *out_public = NULL;

    TSS2_RC rc = Esys_Create(
        g_esys,
        g_primary,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        &in_sensitive,
        &in_public,
        &outside_info,
        &creation_pcr,
        &out_private,
        &out_public,
        NULL,
        NULL,
        NULL
    );

    if (rc != TSS2_RC_SUCCESS) {
        if (out_private) {
            Esys_Free(out_private);
        }
        if (out_public) {
            Esys_Free(out_public);
        }
        return map_tss2_rc(rc);
    }

    rc = Esys_Load(
        g_esys,
        g_primary,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        out_private,
        out_public,
        &g_ak
    );

    if (rc != TSS2_RC_SUCCESS) {
        Esys_Free(out_private);
        Esys_Free(out_public);
        return map_tss2_rc(rc);
    }

    TPM2B_AUTH empty_auth = {.size = 0};
    Esys_TR_SetAuth(g_esys, g_ak, &empty_auth);
    g_ak_public = *out_public;

    Esys_Free(out_private);
    Esys_Free(out_public);
    return TPM2_RC_SUCCESS;
}

static const EVP_MD *map_hash_alg_to_evp(tpm2_crypto_algorithm_t alg) {
    switch (alg) {
        case CRYPTO_ALG_SHA256: return EVP_sha256();
        case CRYPTO_ALG_SHA384: return EVP_sha384();
        case CRYPTO_ALG_SHA512: return EVP_sha512();
#ifdef EVP_sha3_256
        case CRYPTO_ALG_SHA3_256: return EVP_sha3_256();
#endif
#ifdef EVP_sha3_384
        case CRYPTO_ALG_SHA3_384: return EVP_sha3_384();
#endif
        default: return NULL;
    }
}

static const EVP_MD *map_tpm_hash_to_evp(uint16_t alg) {
    switch (alg) {
        case TPM2_ALG_SHA256: return EVP_sha256();
        case TPM2_ALG_SHA384: return EVP_sha384();
        case TPM2_ALG_SHA512: return EVP_sha512();
        default: return NULL;
    }
}

static const EVP_CIPHER *map_cipher_to_evp(tpm2_crypto_algorithm_t alg, size_t key_len) {
    switch (alg) {
        case CRYPTO_ALG_AES_128_CBC:
            return (key_len == 16) ? EVP_aes_128_cbc() : NULL;
        case CRYPTO_ALG_AES_256_CBC:
            return (key_len == 32) ? EVP_aes_256_cbc() : NULL;
        default:
            return NULL;
    }
}

static tpm2_rc_t compute_task_nonce(const char *agent_id, const uint8_t *payload, size_t payload_len, uint8_t *digest_out, size_t *digest_len) {
    if (!agent_id || !digest_out || !digest_len || *digest_len < TPM2_DIGEST_SIZE) {
        return TPM2_RC_BAD_PARAMETER;
    }

    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    if (!md_ctx) {
        return TPM2_RC_FAILURE;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    if (EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    if (EVP_DigestUpdate(md_ctx, agent_id, strlen(agent_id)) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    if (payload && payload_len > 0) {
        if (EVP_DigestUpdate(md_ctx, payload, payload_len) != 1) {
            rc = TPM2_RC_FAILURE;
            goto cleanup;
        }
    }

    unsigned int len = 0;
    if (EVP_DigestFinal_ex(md_ctx, digest_out, &len) != 1) {
        rc = TPM2_RC_FAILURE;
        goto cleanup;
    }

    *digest_len = len;

cleanup:
    EVP_MD_CTX_free(md_ctx);
    return rc;
}

static tpm2_rc_t extend_task_digest_locked(const uint8_t *digest, size_t digest_len) {
    if (digest_len != TPM2_DIGEST_SIZE) {
        return TPM2_RC_BAD_PARAMETER;
    }

    TPML_DIGEST_VALUES digests = {.count = 1};
    digests.digests[0].hashAlg = TPM2_ALG_SHA256;
    memcpy(digests.digests[0].digest.sha256, digest, TPM2_DIGEST_SIZE);

    ESYS_TR pcr_handle = ESYS_TR_PCR0 + TPM2_AGENT_PCR_INDEX;

    TSS2_RC rc = Esys_PCR_Extend(
        g_esys,
        pcr_handle,
        ESYS_TR_PASSWORD,
        ESYS_TR_NONE,
        ESYS_TR_NONE,
        &digests
    );

    return map_tss2_rc(rc);
}

static EVP_PKEY *evp_from_tpm_public(const TPM2B_PUBLIC *public_area) {
    if (!public_area) {
        return NULL;
    }

    if (public_area->publicArea.type != TPM2_ALG_ECC) {
        return NULL;
    }

    int nid = map_tpm_ecc_curve_to_nid(public_area->publicArea.parameters.eccDetail.curveID);
    if (nid == NID_undef) {
        return NULL;
    }

    const TPM2B_ECC_PARAMETER *x = &public_area->publicArea.unique.ecc.x;
    const TPM2B_ECC_PARAMETER *y = &public_area->publicArea.unique.ecc.y;

    BIGNUM *bn_x = BN_bin2bn(x->buffer, x->size, NULL);
    BIGNUM *bn_y = BN_bin2bn(y->buffer, y->size, NULL);
    if (!bn_x || !bn_y) {
        BN_free(bn_x);
        BN_free(bn_y);
        return NULL;
    }

    EC_KEY *ec_key = EC_KEY_new_by_curve_name(nid);
    if (!ec_key) {
        BN_free(bn_x);
        BN_free(bn_y);
        return NULL;
    }

    if (EC_KEY_set_public_key_affine_coordinates(ec_key, bn_x, bn_y) != 1) {
        BN_free(bn_x);
        BN_free(bn_y);
        EC_KEY_free(ec_key);
        return NULL;
    }

    BN_free(bn_x);
    BN_free(bn_y);

    EVP_PKEY *pkey = EVP_PKEY_new();
    if (!pkey) {
        EC_KEY_free(ec_key);
        return NULL;
    }

    if (EVP_PKEY_assign_EC_KEY(pkey, ec_key) != 1) {
        EVP_PKEY_free(pkey);
        EC_KEY_free(ec_key);
        return NULL;
    }

    return pkey;
}

static int map_tpm_ecc_curve_to_nid(TPM2_ECC_CURVE curve) {
    switch (curve) {
        case TPM2_ECC_NIST_P256: return NID_X9_62_prime256v1;
        case TPM2_ECC_NIST_P384: return NID_secp384r1;
        case TPM2_ECC_NIST_P521: return NID_secp521r1;
        default: return NID_undef;
    }
}

static tpm2_rc_t export_attestation_public(uint8_t *buffer, size_t *len) {
    if (!buffer || !len) {
        return TPM2_RC_BAD_PARAMETER;
    }

    EVP_PKEY *pkey = evp_from_tpm_public(&g_ak_public);
    if (!pkey) {
        return TPM2_RC_FAILURE;
    }

    int required = i2d_PUBKEY(pkey, NULL);
    if (required <= 0) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_FAILURE;
    }

    if (*len < (size_t)required) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    unsigned char *tmp = buffer;
    int written = i2d_PUBKEY(pkey, &tmp);
    EVP_PKEY_free(pkey);

    if (written != required) {
        return TPM2_RC_FAILURE;
    }

    *len = (size_t)written;
    return TPM2_RC_SUCCESS;
}

static tpm2_rc_t encode_signature_der(const TPMT_SIGNATURE *signature, uint8_t *buffer, size_t *len) {
    if (!signature || !buffer || !len) {
        return TPM2_RC_BAD_PARAMETER;
    }

    if (signature->sigAlg != TPM2_ALG_ECDSA) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    const TPM2B_ECC_PARAMETER *r = &signature->signature.ecdsa.signatureR;
    const TPM2B_ECC_PARAMETER *s = &signature->signature.ecdsa.signatureS;

    BIGNUM *bn_r = BN_bin2bn(r->buffer, r->size, NULL);
    BIGNUM *bn_s = BN_bin2bn(s->buffer, s->size, NULL);
    if (!bn_r || !bn_s) {
        BN_free(bn_r);
        BN_free(bn_s);
        return TPM2_RC_FAILURE;
    }

    ECDSA_SIG *ecdsa_sig = ECDSA_SIG_new();
    if (!ecdsa_sig) {
        BN_free(bn_r);
        BN_free(bn_s);
        return TPM2_RC_FAILURE;
    }

    if (ECDSA_SIG_set0(ecdsa_sig, bn_r, bn_s) != 1) {
        ECDSA_SIG_free(ecdsa_sig);
        BN_free(bn_r);
        BN_free(bn_s);
        return TPM2_RC_FAILURE;
    }

    int required = i2d_ECDSA_SIG(ecdsa_sig, NULL);
    if (required <= 0) {
        ECDSA_SIG_free(ecdsa_sig);
        return TPM2_RC_FAILURE;
    }

    if (*len < (size_t)required) {
        ECDSA_SIG_free(ecdsa_sig);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    unsigned char *tmp = buffer;
    int written = i2d_ECDSA_SIG(ecdsa_sig, &tmp);
    ECDSA_SIG_free(ecdsa_sig);

    if (written != required) {
        return TPM2_RC_FAILURE;
    }

    *len = (size_t)written;
    return TPM2_RC_SUCCESS;
}

static tpm2_rc_t verify_signature_der(uint16_t hash_alg, const uint8_t *public_key, size_t public_key_len, const uint8_t *signature, size_t signature_len, const uint8_t *data, size_t data_len) {
    if (!public_key || !signature || !data) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_MD *md = map_tpm_hash_to_evp(hash_alg);
    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    const unsigned char *pub_ptr = public_key;
    EVP_PKEY *pkey = d2i_PUBKEY(NULL, &pub_ptr, (long)public_key_len);
    if (!pkey) {
        return TPM2_RC_FAILURE;
    }

    const unsigned char *sig_ptr = signature;
    ECDSA_SIG *ecdsa_sig = d2i_ECDSA_SIG(NULL, &sig_ptr, (long)signature_len);
    if (!ecdsa_sig) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_FAILURE;
    }

    unsigned int digest_len = 0;
    uint8_t digest[EVP_MAX_MD_SIZE];
    if (EVP_Digest(data, data_len, digest, &digest_len, md, NULL) != 1) {
        ECDSA_SIG_free(ecdsa_sig);
        EVP_PKEY_free(pkey);
        return TPM2_RC_FAILURE;
    }

    EC_KEY *ec = EVP_PKEY_get1_EC_KEY(pkey);
    if (!ec) {
        ECDSA_SIG_free(ecdsa_sig);
        EVP_PKEY_free(pkey);
        return TPM2_RC_FAILURE;
    }

    int verified = ECDSA_do_verify(digest, (int)digest_len, ecdsa_sig, ec);
    EC_KEY_free(ec);
    ECDSA_SIG_free(ecdsa_sig);
    EVP_PKEY_free(pkey);

    return (verified == 1) ? TPM2_RC_SUCCESS : TPM2_RC_SECURITY_VIOLATION;
}

static tpm2_rc_t parse_attestation_blob(const uint8_t *blob, size_t blob_len, TPMS_ATTEST *decoded) {
    if (!blob || !decoded) {
        return TPM2_RC_BAD_PARAMETER;
    }

    size_t offset = 0;
    TSS2_RC rc = Tss2_MU_TPMS_ATTEST_Unmarshal(blob, blob_len, &offset, decoded);
    if (rc != TSS2_RC_SUCCESS) {
        return map_tss2_rc(rc);
    }

    return TPM2_RC_SUCCESS;
}

static void clear_agent_state_locked(void) {
    memset(g_active_agent, 0, sizeof(g_active_agent));
    memset(g_active_nonce, 0, sizeof(g_active_nonce));
    g_active_nonce_len = 0;
}
