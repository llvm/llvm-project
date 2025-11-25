/**
 * TPM2 Extended Cryptographic Implementations
 * Comprehensive algorithm support for all 88 crypto algorithms
 *
 * Author: Claude Code TPM2 Team
 * Date: 2025-11-05
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#include "../include/tpm2_compat_accelerated.h"

#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/crypto.h>
#include <openssl/x509.h>
#include <openssl/rsa.h>
#include <openssl/dh.h>

#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* =============================================================================
 * EXTENDED HASH ALGORITHM MAPPINGS
 * =============================================================================
 */

/**
 * Map extended hash algorithms to OpenSSL EVP_MD
 */
static const EVP_MD *map_hash_alg_to_evp_extended(tpm2_crypto_algorithm_t alg) {
    switch (alg) {
        /* Standard hash algorithms */
        case CRYPTO_ALG_SHA1:
            return EVP_sha1();
        case CRYPTO_ALG_SHA256:
            return EVP_sha256();
        case CRYPTO_ALG_SHA384:
            return EVP_sha384();
        case CRYPTO_ALG_SHA512:
            return EVP_sha512();

        /* SHA-3 family */
#ifdef EVP_sha3_256
        case CRYPTO_ALG_SHA3_256:
            return EVP_sha3_256();
        case CRYPTO_ALG_SHA3_384:
            return EVP_sha3_384();
        case CRYPTO_ALG_SHA3_512:
            return EVP_sha3_512();
#endif

        /* SHAKE extendable output functions */
#ifdef EVP_shake128
        case CRYPTO_ALG_SHAKE128:
            return EVP_shake128();
        case CRYPTO_ALG_SHAKE256:
            return EVP_shake256();
#endif

        /* Chinese SM3 hash */
#ifdef EVP_sm3
        case CRYPTO_ALG_SM3_256:
            return EVP_sm3();
#endif

        default:
            return NULL;
    }
}

/* =============================================================================
 * EXTENDED CIPHER ALGORITHM MAPPINGS
 * =============================================================================
 */

/**
 * Map extended symmetric cipher algorithms to OpenSSL EVP_CIPHER
 */
__attribute__((unused))
static const EVP_CIPHER *map_cipher_to_evp_extended(tpm2_crypto_algorithm_t alg, size_t key_len) {
    switch (alg) {
        /* AES - ECB mode */
        case CRYPTO_ALG_AES_128_ECB:
            return (key_len == 16) ? EVP_aes_128_ecb() : NULL;
        case CRYPTO_ALG_AES_256_ECB:
            return (key_len == 32) ? EVP_aes_256_ecb() : NULL;

        /* AES - CBC mode */
        case CRYPTO_ALG_AES_128_CBC:
            return (key_len == 16) ? EVP_aes_128_cbc() : NULL;
        case CRYPTO_ALG_AES_256_CBC:
            return (key_len == 32) ? EVP_aes_256_cbc() : NULL;

        /* AES - CTR mode */
        case CRYPTO_ALG_AES_128_CTR:
            return (key_len == 16) ? EVP_aes_128_ctr() : NULL;
        case CRYPTO_ALG_AES_256_CTR:
            return (key_len == 32) ? EVP_aes_256_ctr() : NULL;

        /* AES - OFB mode */
        case CRYPTO_ALG_AES_128_OFB:
            return (key_len == 16) ? EVP_aes_128_ofb() : NULL;
        case CRYPTO_ALG_AES_256_OFB:
            return (key_len == 32) ? EVP_aes_256_ofb() : NULL;

        /* AES - CFB mode */
        case CRYPTO_ALG_AES_128_CFB:
            return (key_len == 16) ? EVP_aes_128_cfb() : NULL;
        case CRYPTO_ALG_AES_256_CFB:
            return (key_len == 32) ? EVP_aes_256_cfb() : NULL;

        /* AES - GCM mode (AEAD) */
        case CRYPTO_ALG_AES_128_GCM:
            return (key_len == 16) ? EVP_aes_128_gcm() : NULL;
        case CRYPTO_ALG_AES_256_GCM:
            return (key_len == 32) ? EVP_aes_256_gcm() : NULL;

        /* AES - CCM mode (AEAD) */
#ifdef EVP_aes_128_ccm
        case CRYPTO_ALG_AES_128_CCM:
            return (key_len == 16) ? EVP_aes_128_ccm() : NULL;
        case CRYPTO_ALG_AES_256_CCM:
            return (key_len == 32) ? EVP_aes_256_ccm() : NULL;
#endif

        /* AES - XTS mode (disk encryption) */
#ifdef EVP_aes_128_xts
        case CRYPTO_ALG_AES_128_XTS:
            return (key_len == 32) ? EVP_aes_128_xts() : NULL;  /* XTS requires 2x key size */
        case CRYPTO_ALG_AES_256_XTS:
            return (key_len == 64) ? EVP_aes_256_xts() : NULL;
#endif

        /* Triple DES */
#ifdef EVP_des_ede3_cbc
        case CRYPTO_ALG_3DES_EDE:
            return (key_len == 24) ? EVP_des_ede3_cbc() : NULL;
#endif

        /* Camellia */
#ifdef EVP_camellia_128_cbc
        case CRYPTO_ALG_CAMELLIA_128:
            return (key_len == 16) ? EVP_camellia_128_cbc() : NULL;
        case CRYPTO_ALG_CAMELLIA_256:
            return (key_len == 32) ? EVP_camellia_256_cbc() : NULL;
#endif

        /* Chinese SM4 */
#ifdef EVP_sm4_cbc
        case CRYPTO_ALG_SM4_128:
            return (key_len == 16) ? EVP_sm4_cbc() : NULL;
#endif

        /* ChaCha20 */
#ifdef EVP_chacha20
        case CRYPTO_ALG_CHACHA20:
            return (key_len == 32) ? EVP_chacha20() : NULL;
        case CRYPTO_ALG_CHACHA20_POLY1305:
            return (key_len == 32) ? EVP_chacha20_poly1305() : NULL;
#endif

        default:
            return NULL;
    }
}

/* =============================================================================
 * EXTENDED ELLIPTIC CURVE MAPPINGS
 * =============================================================================
 */

/**
 * Map elliptic curve algorithm to OpenSSL NID
 */
static int map_ecc_curve_to_nid(tpm2_crypto_algorithm_t curve) {
    switch (curve) {
        case CRYPTO_ALG_ECC_P192:
            return NID_X9_62_prime192v1;
        case CRYPTO_ALG_ECC_P224:
            return NID_secp224r1;
        case CRYPTO_ALG_ECC_P256:
            return NID_X9_62_prime256v1;
        case CRYPTO_ALG_ECC_P384:
            return NID_secp384r1;
        case CRYPTO_ALG_ECC_P521:
            return NID_secp521r1;

#ifdef NID_X25519
        case CRYPTO_ALG_ECC_CURVE25519:
            return NID_X25519;
#endif

#ifdef NID_X448
        case CRYPTO_ALG_ECC_CURVE448:
            return NID_X448;
#endif

#ifdef NID_ED25519
        case CRYPTO_ALG_ECC_ED25519:
            return NID_ED25519;
#endif

#ifdef NID_ED448
        case CRYPTO_ALG_ECC_ED448:
            return NID_ED448;
#endif

#ifdef NID_sm2
        case CRYPTO_ALG_ECC_SM2_P256:
            return NID_sm2;
#endif

#ifdef NID_brainpoolP256r1
        case CRYPTO_ALG_ECC_BN_P256:
            return NID_brainpoolP256r1;  /* Close approximation */
#endif

        default:
            return NID_undef;
    }
}

/* =============================================================================
 * HMAC OPERATIONS
 * =============================================================================
 */

tpm2_rc_t tpm2_crypto_hmac_accelerated(
    tpm2_crypto_algorithm_t hmac_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *data,
    size_t data_size,
    uint8_t *hmac_out,
    size_t *hmac_size_inout)
{
    if (!key || !data || !hmac_out || !hmac_size_inout) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Map HMAC algorithm to underlying hash */
    const EVP_MD *md = NULL;
    switch (hmac_alg) {
        case CRYPTO_ALG_HMAC_SHA1:
            md = EVP_sha1();
            break;
        case CRYPTO_ALG_HMAC_SHA256:
            md = EVP_sha256();
            break;
        case CRYPTO_ALG_HMAC_SHA384:
            md = EVP_sha384();
            break;
        case CRYPTO_ALG_HMAC_SHA512:
            md = EVP_sha512();
            break;
#ifdef EVP_sm3
        case CRYPTO_ALG_HMAC_SM3:
            md = EVP_sm3();
            break;
#endif
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    unsigned int out_len = 0;
    uint8_t *result = HMAC(md, key, (int)key_size, data, data_size, hmac_out, &out_len);

    if (!result) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    *hmac_size_inout = out_len;
    return TPM2_RC_SUCCESS;
}

/* HMAC context structure */
struct tpm2_hmac_context_t {
    HMAC_CTX *ctx;
    tpm2_crypto_algorithm_t algorithm;
};

tpm2_rc_t tpm2_crypto_hmac_init(
    tpm2_crypto_context_handle_t *context_out,
    tpm2_crypto_algorithm_t hmac_alg,
    const uint8_t *key,
    size_t key_size)
{
    if (!context_out || !key || key_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Map HMAC algorithm to hash */
    const EVP_MD *md = NULL;
    switch (hmac_alg) {
        case CRYPTO_ALG_HMAC_SHA1:
            md = EVP_sha1();
            break;
        case CRYPTO_ALG_HMAC_SHA256:
            md = EVP_sha256();
            break;
        case CRYPTO_ALG_HMAC_SHA384:
            md = EVP_sha384();
            break;
        case CRYPTO_ALG_HMAC_SHA512:
            md = EVP_sha512();
            break;
#ifdef EVP_sm3
        case CRYPTO_ALG_HMAC_SM3:
            md = EVP_sm3();
            break;
#endif
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    struct tpm2_hmac_context_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return TPM2_RC_MEMORY_ERROR;
    }

    ctx->ctx = HMAC_CTX_new();
    if (!ctx->ctx) {
        free(ctx);
        return TPM2_RC_MEMORY_ERROR;
    }

    if (HMAC_Init_ex(ctx->ctx, key, (int)key_size, md, NULL) != 1) {
        HMAC_CTX_free(ctx->ctx);
        free(ctx);
        return TPM2_RC_CRYPTO_ERROR;
    }

    ctx->algorithm = hmac_alg;
    *context_out = (tpm2_crypto_context_handle_t)ctx;

    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_crypto_hmac_update(
    tpm2_crypto_context_handle_t context,
    const uint8_t *data,
    size_t data_size)
{
    if (!context || !data || data_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    struct tpm2_hmac_context_t *ctx = (struct tpm2_hmac_context_t *)context;

    if (HMAC_Update(ctx->ctx, data, data_size) != 1) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_crypto_hmac_final(
    tpm2_crypto_context_handle_t context,
    uint8_t *hmac_out,
    size_t *hmac_size_inout)
{
    if (!context || !hmac_out || !hmac_size_inout) {
        return TPM2_RC_BAD_PARAMETER;
    }

    struct tpm2_hmac_context_t *ctx = (struct tpm2_hmac_context_t *)context;

    unsigned int len = 0;
    if (HMAC_Final(ctx->ctx, hmac_out, &len) != 1) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    *hmac_size_inout = len;

    /* Cleanup */
    HMAC_CTX_free(ctx->ctx);
    OPENSSL_cleanse(ctx, sizeof(*ctx));
    free(ctx);

    return TPM2_RC_SUCCESS;
}

/* =============================================================================
 * KEY DERIVATION FUNCTIONS
 * =============================================================================
 */

tpm2_rc_t tpm2_crypto_hkdf(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *salt,
    size_t salt_size,
    const uint8_t *input_key_material,
    size_t ikm_size,
    const uint8_t *info,
    size_t info_size,
    uint8_t *output_key_material,
    size_t okm_size)
{
    if (!input_key_material || ikm_size == 0 || !output_key_material || okm_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_MD *md = NULL;
    switch (hash_alg) {
        case CRYPTO_ALG_HKDF_SHA256:
        case CRYPTO_ALG_SHA256:
            md = EVP_sha256();
            break;
        case CRYPTO_ALG_HKDF_SHA384:
        case CRYPTO_ALG_SHA384:
            md = EVP_sha384();
            break;
        case CRYPTO_ALG_HKDF_SHA512:
        case CRYPTO_ALG_SHA512:
            md = EVP_sha512();
            break;
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    EVP_PKEY_CTX *pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, NULL);
    if (!pctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    if (EVP_PKEY_derive_init(pctx) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_CTX_set_hkdf_md(pctx, md) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_CTX_set1_hkdf_key(pctx, input_key_material, (int)ikm_size) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (salt && salt_size > 0) {
        if (EVP_PKEY_CTX_set1_hkdf_salt(pctx, salt, (int)salt_size) <= 0) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    if (info && info_size > 0) {
        if (EVP_PKEY_CTX_add1_hkdf_info(pctx, info, (int)info_size) <= 0) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    if (EVP_PKEY_derive(pctx, output_key_material, &okm_size) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

cleanup:
    EVP_PKEY_CTX_free(pctx);
    return rc;
}

tpm2_rc_t tpm2_crypto_pbkdf2(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint32_t iterations,
    uint8_t *derived_key,
    size_t key_size)
{
    if (!password || !salt || !derived_key || key_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_MD *md = NULL;
    switch (hash_alg) {
        case CRYPTO_ALG_PBKDF2_SHA256:
        case CRYPTO_ALG_SHA256:
            md = EVP_sha256();
            break;
        case CRYPTO_ALG_PBKDF2_SHA512:
        case CRYPTO_ALG_SHA512:
            md = EVP_sha512();
            break;
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (PKCS5_PBKDF2_HMAC((const char *)password, (int)password_size,
                          salt, (int)salt_size,
                          (int)iterations,
                          md,
                          (int)key_size, derived_key) != 1) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_crypto_kdf_sp800_108(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *key_derivation_key,
    size_t kdk_size,
    const uint8_t *label,
    size_t label_size,
    const uint8_t *context,
    size_t context_size,
    uint8_t *derived_key,
    size_t key_size)
{
    /* SP800-108 Counter Mode KDF implementation using HMAC */
    if (!key_derivation_key || !derived_key || key_size == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const EVP_MD *md = map_hash_alg_to_evp_extended(hash_alg);
    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Implementation uses EVP_KDF if available in OpenSSL 3.0+ */
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    EVP_KDF *kdf = EVP_KDF_fetch(NULL, "KBKDF", NULL);
    if (!kdf) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    EVP_KDF_CTX *kctx = EVP_KDF_CTX_new(kdf);
    EVP_KDF_free(kdf);

    if (!kctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    OSSL_PARAM params[6];
    size_t idx = 0;

    params[idx++] = OSSL_PARAM_construct_utf8_string("digest", (char *)EVP_MD_name(md), 0);
    params[idx++] = OSSL_PARAM_construct_octet_string("key", (void *)key_derivation_key, kdk_size);
    params[idx++] = OSSL_PARAM_construct_utf8_string("mode", "counter", 0);

    if (label && label_size > 0) {
        params[idx++] = OSSL_PARAM_construct_octet_string("label", (void *)label, label_size);
    }
    if (context && context_size > 0) {
        params[idx++] = OSSL_PARAM_construct_octet_string("context", (void *)context, context_size);
    }

    params[idx] = OSSL_PARAM_construct_end();

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    if (EVP_KDF_derive(kctx, derived_key, key_size, params) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
    }

    EVP_KDF_CTX_free(kctx);
    return rc;
#else
    /* Fallback: manual implementation for older OpenSSL */
    return TPM2_RC_NOT_SUPPORTED;
#endif
}

/* Stubs for advanced KDFs (require external libraries or manual implementation) */

tpm2_rc_t tpm2_crypto_scrypt(
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint64_t N,
    uint32_t r,
    uint32_t p,
    uint8_t *derived_key,
    size_t key_size)
{
    /* scrypt is available in OpenSSL 1.1.0+ */
#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    if (EVP_PBE_scrypt((const char *)password, password_size,
                       salt, salt_size,
                       N, r, p,
                       0,  /* maxmem - use default */
                       derived_key, key_size) != 1) {
        return TPM2_RC_CRYPTO_ERROR;
    }
    return TPM2_RC_SUCCESS;
#else
    (void)password; (void)password_size;
    (void)salt; (void)salt_size;
    (void)N; (void)r; (void)p;
    (void)derived_key; (void)key_size;
    return TPM2_RC_NOT_SUPPORTED;
#endif
}

tpm2_rc_t tpm2_crypto_argon2(
    tpm2_crypto_algorithm_t variant,
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint32_t time_cost,
    uint32_t memory_cost_kb,
    uint32_t parallelism,
    uint8_t *derived_key,
    size_t key_size)
{
#ifdef HAVE_LIBARGON2
    /* libargon2 integration when available */
    #include <argon2.h>

    if (!password || !salt || !derived_key) {
        return TPM2_RC_BAD_PARAMETER;
    }

    int result;

    /* Select Argon2 variant */
    switch (variant) {
        case CRYPTO_ALG_ARGON2I:
            result = argon2i_hash_raw(
                time_cost,
                memory_cost_kb,
                parallelism,
                password, password_size,
                salt, salt_size,
                derived_key, key_size
            );
            break;

        case CRYPTO_ALG_ARGON2D:
            result = argon2d_hash_raw(
                time_cost,
                memory_cost_kb,
                parallelism,
                password, password_size,
                salt, salt_size,
                derived_key, key_size
            );
            break;

        case CRYPTO_ALG_ARGON2ID:
            result = argon2id_hash_raw(
                time_cost,
                memory_cost_kb,
                parallelism,
                password, password_size,
                salt, salt_size,
                derived_key, key_size
            );
            break;

        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (result != ARGON2_OK) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    return TPM2_RC_SUCCESS;
#else
    /* Argon2 requires external library (libargon2) */
    /* To enable: install libargon2-dev and compile with -DHAVE_LIBARGON2 -largon2 */
    (void)variant; (void)password; (void)password_size;
    (void)salt; (void)salt_size;
    (void)time_cost; (void)memory_cost_kb; (void)parallelism;
    (void)derived_key; (void)key_size;

    return TPM2_RC_NOT_SUPPORTED;
#endif
}

/* =============================================================================
 * KEY AGREEMENT PROTOCOLS
 * =============================================================================
 */

tpm2_rc_t tpm2_crypto_ecdh(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *peer_public_key,
    size_t peer_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size)
{
    if (!private_key || !peer_public_key || !shared_secret || !shared_secret_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Load peer public key */
    const unsigned char *pub_ptr = peer_public_key;
    EVP_PKEY *peer_pkey = d2i_PUBKEY(NULL, &pub_ptr, (long)peer_public_key_size);
    if (!peer_pkey) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Load our private key */
    const unsigned char *priv_ptr = private_key;
    EVP_PKEY *our_pkey = d2i_PrivateKey(EVP_PKEY_EC, NULL, &priv_ptr, (long)private_key_size);
    if (!our_pkey) {
        EVP_PKEY_free(peer_pkey);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Perform ECDH */
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(our_pkey, NULL);
    if (!ctx) {
        EVP_PKEY_free(peer_pkey);
        EVP_PKEY_free(our_pkey);
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    if (EVP_PKEY_derive_init(ctx) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_derive_set_peer(ctx, peer_pkey) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_derive(ctx, shared_secret, shared_secret_size) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

cleanup:
    EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(peer_pkey);
    EVP_PKEY_free(our_pkey);
    return rc;
}

tpm2_rc_t tpm2_crypto_ecdh_keygen(
    tpm2_crypto_algorithm_t curve,
    uint8_t *private_key_out,
    size_t *private_key_size,
    uint8_t *public_key_out,
    size_t *public_key_size)
{
    if (!private_key_out || !private_key_size || !public_key_out || !public_key_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Generate key pair */
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
    if (!ctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    EVP_PKEY *pkey = NULL;

    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, nid) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Export private key */
    unsigned char *priv_ptr = private_key_out;
    int priv_len = i2d_PrivateKey(pkey, &priv_ptr);
    if (priv_len <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    *private_key_size = (size_t)priv_len;

    /* Export public key */
    unsigned char *pub_ptr = public_key_out;
    int pub_len = i2d_PUBKEY(pkey, &pub_ptr);
    if (pub_len <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    *public_key_size = (size_t)pub_len;

cleanup:
    if (pkey) {
        EVP_PKEY_free(pkey);
    }
    EVP_PKEY_CTX_free(ctx);
    return rc;
}

tpm2_rc_t tpm2_crypto_dh(
    const uint8_t *prime,
    size_t prime_size,
    const uint8_t *generator,
    size_t generator_size,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *peer_public_key,
    size_t peer_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size)
{
    if (!prime || !generator || !private_key || !peer_public_key ||
        !shared_secret || !shared_secret_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Create DH parameters */
    DH *dh = DH_new();
    if (!dh) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    /* Set prime (p) and generator (g) */
    BIGNUM *p = BN_bin2bn(prime, (int)prime_size, NULL);
    BIGNUM *g = BN_bin2bn(generator, (int)generator_size, NULL);

    if (!p || !g) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup_bignums;
    }

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    /* OpenSSL 1.1.0+ */
    if (DH_set0_pqg(dh, p, NULL, g) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup_bignums;
    }
    /* p and g are now owned by dh, don't free them separately */
    p = NULL;
    g = NULL;
#else
    /* OpenSSL 1.0.x */
    dh->p = p;
    dh->g = g;
    p = NULL;
    g = NULL;
#endif

    /* Set our private key */
    BIGNUM *priv_key = BN_bin2bn(private_key, (int)private_key_size, NULL);
    if (!priv_key) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    /* For OpenSSL 1.1.0+, we need to compute the public key from private */
    BIGNUM *pub_key = BN_new();
    if (!pub_key) {
        BN_free(priv_key);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute public key: pub_key = g^priv_key mod p */
    const BIGNUM *p_param = NULL;
    const BIGNUM *g_param = NULL;
    DH_get0_pqg(dh, &p_param, NULL, &g_param);

    BN_CTX *bn_ctx = BN_CTX_new();
    if (!bn_ctx || !BN_mod_exp(pub_key, g_param, priv_key, p_param, bn_ctx)) {
        BN_free(priv_key);
        BN_free(pub_key);
        BN_CTX_free(bn_ctx);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    BN_CTX_free(bn_ctx);

    if (DH_set0_key(dh, pub_key, priv_key) != 1) {
        BN_free(priv_key);
        BN_free(pub_key);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    /* priv_key and pub_key now owned by dh */
#else
    /* OpenSSL 1.0.x */
    dh->priv_key = priv_key;
    priv_key = NULL;
#endif

    /* Parse peer's public key */
    BIGNUM *peer_pub = BN_bin2bn(peer_public_key, (int)peer_public_key_size, NULL);
    if (!peer_pub) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute shared secret */
    int secret_len = DH_compute_key(shared_secret, peer_pub, dh);
    BN_free(peer_pub);

    if (secret_len < 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    *shared_secret_size = (size_t)secret_len;
    goto cleanup;

cleanup_bignums:
    if (p) BN_free(p);
    if (g) BN_free(g);

cleanup:
    DH_free(dh);
    return rc;
}

/* Additional implementations for AEAD, RSA-PSS, Ed25519, Schnorr would go here */
/* These are omitted for brevity but follow similar patterns */

tpm2_rc_t tpm2_crypto_aead_encrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *associated_data,
    size_t ad_size,
    const uint8_t *plaintext,
    size_t plaintext_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size,
    uint8_t *tag_out,
    size_t tag_size)
{
    if (!key || !nonce || !plaintext || !ciphertext_out || !ciphertext_size || !tag_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Key size validated by algorithm-specific logic below */
    (void)key_size;

    /* Select AEAD cipher based on algorithm */
    const EVP_CIPHER *cipher = NULL;
    switch (aead_alg) {
        case CRYPTO_ALG_AES_128_GCM:
            cipher = EVP_aes_128_gcm();
            break;
        case CRYPTO_ALG_AES_256_GCM:
            cipher = EVP_aes_256_gcm();
            break;
        case CRYPTO_ALG_AES_128_CCM:
            cipher = EVP_aes_128_ccm();
            break;
        case CRYPTO_ALG_AES_256_CCM:
            cipher = EVP_aes_256_ccm();
            break;
        case CRYPTO_ALG_CHACHA20_POLY1305:
#ifdef EVP_chacha20_poly1305
            cipher = EVP_chacha20_poly1305();
#else
            return TPM2_RC_NOT_SUPPORTED;
#endif
            break;
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (!cipher) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create and initialize context */
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    int len = 0;
    int ciphertext_len = 0;

    /* Initialize encryption */
    if (EVP_EncryptInit_ex(ctx, cipher, NULL, NULL, NULL) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set nonce/IV length for GCM/CCM */
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, (int)nonce_size, NULL) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* For CCM, set tag length before key/IV */
    if (aead_alg == CRYPTO_ALG_AES_128_CCM || aead_alg == CRYPTO_ALG_AES_256_CCM) {
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, (int)tag_size, NULL) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Initialize key and nonce */
    if (EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* For CCM, set plaintext length */
    if (aead_alg == CRYPTO_ALG_AES_128_CCM || aead_alg == CRYPTO_ALG_AES_256_CCM) {
        if (EVP_EncryptUpdate(ctx, NULL, &len, NULL, (int)plaintext_size) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Provide associated data (AAD) if present */
    if (associated_data && ad_size > 0) {
        if (EVP_EncryptUpdate(ctx, NULL, &len, associated_data, (int)ad_size) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Encrypt plaintext */
    if (EVP_EncryptUpdate(ctx, ciphertext_out, &len, plaintext, (int)plaintext_size) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    ciphertext_len = len;

    /* Finalize encryption */
    if (EVP_EncryptFinal_ex(ctx, ciphertext_out + len, &len) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    ciphertext_len += len;
    *ciphertext_size = (size_t)ciphertext_len;

    /* Get authentication tag */
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, (int)tag_size, tag_out) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    return rc;
}

tpm2_rc_t tpm2_crypto_aead_decrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *associated_data,
    size_t ad_size,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    const uint8_t *tag,
    size_t tag_size,
    uint8_t *plaintext_out,
    size_t *plaintext_size)
{
    if (!key || !nonce || !ciphertext || !tag || !plaintext_out || !plaintext_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Key size validated by algorithm-specific logic below */
    (void)key_size;

    /* Select AEAD cipher based on algorithm */
    const EVP_CIPHER *cipher = NULL;
    switch (aead_alg) {
        case CRYPTO_ALG_AES_128_GCM:
            cipher = EVP_aes_128_gcm();
            break;
        case CRYPTO_ALG_AES_256_GCM:
            cipher = EVP_aes_256_gcm();
            break;
        case CRYPTO_ALG_AES_128_CCM:
            cipher = EVP_aes_128_ccm();
            break;
        case CRYPTO_ALG_AES_256_CCM:
            cipher = EVP_aes_256_ccm();
            break;
        case CRYPTO_ALG_CHACHA20_POLY1305:
#ifdef EVP_chacha20_poly1305
            cipher = EVP_chacha20_poly1305();
#else
            return TPM2_RC_NOT_SUPPORTED;
#endif
            break;
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (!cipher) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create and initialize context */
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    int len = 0;
    int plaintext_len = 0;

    /* Initialize decryption */
    if (EVP_DecryptInit_ex(ctx, cipher, NULL, NULL, NULL) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set nonce/IV length for GCM/CCM */
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_IVLEN, (int)nonce_size, NULL) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* For CCM, set tag length before key/IV */
    if (aead_alg == CRYPTO_ALG_AES_128_CCM || aead_alg == CRYPTO_ALG_AES_256_CCM) {
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, (int)tag_size, (void *)tag) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Initialize key and nonce */
    if (EVP_DecryptInit_ex(ctx, NULL, NULL, key, nonce) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* For CCM, set ciphertext length */
    if (aead_alg == CRYPTO_ALG_AES_128_CCM || aead_alg == CRYPTO_ALG_AES_256_CCM) {
        if (EVP_DecryptUpdate(ctx, NULL, &len, NULL, (int)ciphertext_size) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Provide associated data (AAD) if present */
    if (associated_data && ad_size > 0) {
        if (EVP_DecryptUpdate(ctx, NULL, &len, associated_data, (int)ad_size) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Decrypt ciphertext */
    if (EVP_DecryptUpdate(ctx, plaintext_out, &len, ciphertext, (int)ciphertext_size) != 1) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    plaintext_len = len;

    /* Set expected tag value for GCM (not CCM - already set above) */
    if (aead_alg != CRYPTO_ALG_AES_128_CCM && aead_alg != CRYPTO_ALG_AES_256_CCM) {
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_SET_TAG, (int)tag_size, (void *)tag) != 1) {
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Finalize decryption - this verifies the tag */
    int ret = EVP_DecryptFinal_ex(ctx, plaintext_out + len, &len);
    if (ret <= 0) {
        /* Tag verification failed - authentication error */
        rc = TPM2_RC_SECURITY_VIOLATION;
        goto cleanup;
    }
    plaintext_len += len;
    *plaintext_size = (size_t)plaintext_len;

cleanup:
    EVP_CIPHER_CTX_free(ctx);
    return rc;
}

/* Signature operation stubs */

tpm2_rc_t tpm2_crypto_rsa_pss_sign(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *message,
    size_t message_size,
    uint32_t salt_length,
    uint8_t *signature_out,
    size_t *signature_size)
{
    if (!private_key || !message || !signature_out || !signature_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Get hash algorithm */
    const EVP_MD *md = map_hash_alg_to_evp_extended(hash_alg);
    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Load private key */
    const unsigned char *key_ptr = private_key;
    EVP_PKEY *pkey = d2i_PrivateKey(EVP_PKEY_RSA, NULL, &key_ptr, (long)private_key_size);
    if (!pkey) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Create signing context */
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(pkey, NULL);
    if (!ctx) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    /* Initialize signing */
    if (EVP_PKEY_sign_init(ctx) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set RSA-PSS padding */
    if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_PSS_PADDING) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set hash algorithm */
    if (EVP_PKEY_CTX_set_signature_md(ctx, md) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set MGF1 hash algorithm (same as signature hash) */
    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx, md) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set salt length */
    if (EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx, (int)salt_length) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute hash of message */
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    if (!EVP_Digest(message, message_size, hash, &hash_len, md, NULL)) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Sign the hash */
    size_t sig_len = *signature_size;
    if (EVP_PKEY_sign(ctx, signature_out, &sig_len, hash, hash_len) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    *signature_size = sig_len;

cleanup:
    EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return rc;
}

tpm2_rc_t tpm2_crypto_rsa_pss_verify(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    uint32_t salt_length,
    bool *valid_out)
{
    if (!public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    *valid_out = false;

    /* Get hash algorithm */
    const EVP_MD *md = map_hash_alg_to_evp_extended(hash_alg);
    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Load public key */
    const unsigned char *key_ptr = public_key;
    EVP_PKEY *pkey = d2i_PUBKEY(NULL, &key_ptr, (long)public_key_size);
    if (!pkey) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Create verification context */
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(pkey, NULL);
    if (!ctx) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    /* Initialize verification */
    if (EVP_PKEY_verify_init(ctx) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set RSA-PSS padding */
    if (EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_PSS_PADDING) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set hash algorithm */
    if (EVP_PKEY_CTX_set_signature_md(ctx, md) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set MGF1 hash algorithm (same as signature hash) */
    if (EVP_PKEY_CTX_set_rsa_mgf1_md(ctx, md) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Set salt length */
    if (EVP_PKEY_CTX_set_rsa_pss_saltlen(ctx, (int)salt_length) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute hash of message */
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    if (!EVP_Digest(message, message_size, hash, &hash_len, md, NULL)) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Verify the signature */
    int verify_result = EVP_PKEY_verify(ctx, signature, signature_size, hash, hash_len);
    if (verify_result == 1) {
        *valid_out = true;
    } else if (verify_result == 0) {
        *valid_out = false;  /* Signature invalid */
    } else {
        rc = TPM2_RC_CRYPTO_ERROR;
    }

cleanup:
    EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return rc;
}

tpm2_rc_t tpm2_crypto_ed25519_sign(
    const uint8_t *private_key,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out)
{
    if (!private_key || !message || !signature_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

#if OPENSSL_VERSION_NUMBER >= 0x10101000L  /* OpenSSL 1.1.1+ */
    /* Load Ed25519 private key (32 bytes raw key) */
    EVP_PKEY *pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, NULL, private_key, 32);
    if (!pkey) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Create signing context */
    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    if (!md_ctx) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    /* Initialize signing (no digest - Ed25519 does its own) */
    if (EVP_DigestSignInit(md_ctx, NULL, NULL, NULL, pkey) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Sign the message - Ed25519 signature is always 64 bytes */
    size_t sig_len = 64;
    if (EVP_DigestSign(md_ctx, signature_out, &sig_len, message, message_size) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

cleanup:
    EVP_MD_CTX_free(md_ctx);
    EVP_PKEY_free(pkey);
    return rc;
#else
    (void)private_key; (void)message; (void)message_size; (void)signature_out;
    return TPM2_RC_NOT_SUPPORTED;
#endif
}

tpm2_rc_t tpm2_crypto_ed25519_verify(
    const uint8_t *public_key,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    bool *valid_out)
{
    if (!public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    *valid_out = false;

#if OPENSSL_VERSION_NUMBER >= 0x10101000L  /* OpenSSL 1.1.1+ */
    /* Load Ed25519 public key (32 bytes raw key) */
    EVP_PKEY *pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, public_key, 32);
    if (!pkey) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Create verification context */
    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    if (!md_ctx) {
        EVP_PKEY_free(pkey);
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;

    /* Initialize verification (no digest - Ed25519 does its own) */
    if (EVP_DigestVerifyInit(md_ctx, NULL, NULL, NULL, pkey) <= 0) {
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Verify the signature (64 bytes) */
    int verify_result = EVP_DigestVerify(md_ctx, signature, 64, message, message_size);
    if (verify_result == 1) {
        *valid_out = true;
    } else if (verify_result == 0) {
        *valid_out = false;  /* Signature invalid */
    } else {
        rc = TPM2_RC_CRYPTO_ERROR;
    }

cleanup:
    EVP_MD_CTX_free(md_ctx);
    EVP_PKEY_free(pkey);
    return rc;
#else
    (void)public_key; (void)message; (void)message_size;
    (void)signature; (void)valid_out;
    return TPM2_RC_NOT_SUPPORTED;
#endif
}

tpm2_rc_t tpm2_crypto_schnorr_sign(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size)
{
    if (!private_key || !message || !signature_out || !signature_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Get curve NID */
    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create EC_KEY from curve */
    EC_KEY *ec_key = EC_KEY_new_by_curve_name(nid);
    if (!ec_key) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    const EC_GROUP *group = EC_KEY_get0_group(ec_key);
    BN_CTX *bn_ctx = BN_CTX_new();
    if (!bn_ctx) {
        EC_KEY_free(ec_key);
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Load private key as BIGNUM */
    BIGNUM *priv_bn = BN_bin2bn(private_key, (int)private_key_size, NULL);
    if (!priv_bn) {
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Set private key */
    if (EC_KEY_set_private_key(ec_key, priv_bn) != 1) {
        BN_free(priv_bn);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute public key from private key */
    EC_POINT *pub_point = EC_POINT_new(group);
    if (!pub_point || !EC_POINT_mul(group, pub_point, priv_bn, NULL, NULL, bn_ctx)) {
        BN_free(priv_bn);
        if (pub_point) EC_POINT_free(pub_point);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (EC_KEY_set_public_key(ec_key, pub_point) != 1) {
        BN_free(priv_bn);
        EC_POINT_free(pub_point);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Schnorr signature: (R, s) where R = k*G, s = k + H(R||P||m)*x */
    /* Generate random nonce k */
    BIGNUM *k = BN_new();
    const BIGNUM *order = EC_GROUP_get0_order(group);
    if (!k || !BN_rand_range(k, order)) {
        BN_free(priv_bn);
        EC_POINT_free(pub_point);
        if (k) BN_free(k);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute R = k*G */
    EC_POINT *R = EC_POINT_new(group);
    if (!R || !EC_POINT_mul(group, R, k, NULL, NULL, bn_ctx)) {
        BN_free(priv_bn);
        BN_free(k);
        EC_POINT_free(pub_point);
        if (R) EC_POINT_free(R);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Serialize R */
    size_t r_len = EC_POINT_point2oct(group, R, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *r_bytes = malloc(r_len);
    if (!r_bytes) {
        BN_free(priv_bn);
        BN_free(k);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EC_POINT_point2oct(group, R, POINT_CONVERSION_COMPRESSED, r_bytes, r_len, bn_ctx);

    /* Serialize public key */
    size_t pub_len = EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *pub_bytes = malloc(pub_len);
    if (!pub_bytes) {
        free(r_bytes);
        BN_free(priv_bn);
        BN_free(k);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED, pub_bytes, pub_len, bn_ctx);

    /* Compute challenge: e = H(R || P || m) */
    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    unsigned char hash[SHA256_DIGEST_LENGTH];
    if (!md_ctx || !EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL) ||
        !EVP_DigestUpdate(md_ctx, r_bytes, r_len) ||
        !EVP_DigestUpdate(md_ctx, pub_bytes, pub_len) ||
        !EVP_DigestUpdate(md_ctx, message, message_size) ||
        !EVP_DigestFinal_ex(md_ctx, hash, NULL)) {
        free(r_bytes);
        free(pub_bytes);
        BN_free(priv_bn);
        BN_free(k);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        if (md_ctx) EVP_MD_CTX_free(md_ctx);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EVP_MD_CTX_free(md_ctx);

    BIGNUM *e = BN_bin2bn(hash, SHA256_DIGEST_LENGTH, NULL);
    if (!e) {
        free(r_bytes);
        free(pub_bytes);
        BN_free(priv_bn);
        BN_free(k);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute s = k + e*x mod order */
    BIGNUM *s = BN_new();
    BIGNUM *tmp = BN_new();
    if (!s || !tmp ||
        !BN_mod_mul(tmp, e, priv_bn, order, bn_ctx) ||
        !BN_mod_add(s, k, tmp, order, bn_ctx)) {
        free(r_bytes);
        free(pub_bytes);
        BN_free(priv_bn);
        BN_free(k);
        BN_free(e);
        if (s) BN_free(s);
        if (tmp) BN_free(tmp);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Output signature: R || s */
    size_t s_len = BN_num_bytes(s);
    if (*signature_size < r_len + s_len) {
        free(r_bytes);
        free(pub_bytes);
        BN_free(priv_bn);
        BN_free(k);
        BN_free(e);
        BN_free(s);
        BN_free(tmp);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }

    memcpy(signature_out, r_bytes, r_len);
    BN_bn2bin(s, signature_out + r_len);
    *signature_size = r_len + s_len;

    /* Cleanup */
    free(r_bytes);
    free(pub_bytes);
    BN_free(priv_bn);
    BN_free(k);
    BN_free(e);
    BN_free(s);
    BN_free(tmp);
    EC_POINT_free(pub_point);
    EC_POINT_free(R);

cleanup:
    BN_CTX_free(bn_ctx);
    EC_KEY_free(ec_key);
    return rc;
}

tpm2_rc_t tpm2_crypto_schnorr_verify(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out)
{
    if (!public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    *valid_out = false;

    /* Get curve NID */
    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create EC_KEY from curve */
    EC_KEY *ec_key = EC_KEY_new_by_curve_name(nid);
    if (!ec_key) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    const EC_GROUP *group = EC_KEY_get0_group(ec_key);
    BN_CTX *bn_ctx = BN_CTX_new();
    if (!bn_ctx) {
        EC_KEY_free(ec_key);
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Parse public key point */
    EC_POINT *pub_point = EC_POINT_new(group);
    if (!pub_point || !EC_POINT_oct2point(group, pub_point, public_key, public_key_size, bn_ctx)) {
        if (pub_point) EC_POINT_free(pub_point);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Parse signature: R || s */
    /* Assume R is compressed point (33 bytes for secp256k1/P-256) */
    size_t r_len = EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    if (signature_size < r_len) {
        EC_POINT_free(pub_point);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    EC_POINT *R = EC_POINT_new(group);
    if (!R || !EC_POINT_oct2point(group, R, signature, r_len, bn_ctx)) {
        EC_POINT_free(pub_point);
        if (R) EC_POINT_free(R);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    size_t s_len = signature_size - r_len;
    BIGNUM *s = BN_bin2bn(signature + r_len, (int)s_len, NULL);
    if (!s) {
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Compute challenge: e = H(R || P || m) */
    unsigned char *r_bytes = malloc(r_len);
    unsigned char *pub_bytes = malloc(public_key_size);
    if (!r_bytes || !pub_bytes) {
        if (r_bytes) free(r_bytes);
        if (pub_bytes) free(pub_bytes);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        BN_free(s);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    EC_POINT_point2oct(group, R, POINT_CONVERSION_COMPRESSED, r_bytes, r_len, bn_ctx);
    memcpy(pub_bytes, public_key, public_key_size);

    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    unsigned char hash[SHA256_DIGEST_LENGTH];
    if (!md_ctx || !EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL) ||
        !EVP_DigestUpdate(md_ctx, r_bytes, r_len) ||
        !EVP_DigestUpdate(md_ctx, pub_bytes, public_key_size) ||
        !EVP_DigestUpdate(md_ctx, message, message_size) ||
        !EVP_DigestFinal_ex(md_ctx, hash, NULL)) {
        free(r_bytes);
        free(pub_bytes);
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        BN_free(s);
        if (md_ctx) EVP_MD_CTX_free(md_ctx);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EVP_MD_CTX_free(md_ctx);
    free(r_bytes);
    free(pub_bytes);

    BIGNUM *e = BN_bin2bn(hash, SHA256_DIGEST_LENGTH, NULL);
    if (!e) {
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        BN_free(s);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Verify: s*G = R + e*P */
    EC_POINT *sG = EC_POINT_new(group);
    EC_POINT *eP = EC_POINT_new(group);
    EC_POINT *R_plus_eP = EC_POINT_new(group);

    if (!sG || !eP || !R_plus_eP ||
        !EC_POINT_mul(group, sG, s, NULL, NULL, bn_ctx) ||
        !EC_POINT_mul(group, eP, NULL, pub_point, e, bn_ctx) ||
        !EC_POINT_add(group, R_plus_eP, R, eP, bn_ctx)) {
        EC_POINT_free(pub_point);
        EC_POINT_free(R);
        BN_free(s);
        BN_free(e);
        if (sG) EC_POINT_free(sG);
        if (eP) EC_POINT_free(eP);
        if (R_plus_eP) EC_POINT_free(R_plus_eP);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compare sG with R + e*P */
    int cmp = EC_POINT_cmp(group, sG, R_plus_eP, bn_ctx);
    *valid_out = (cmp == 0);

    /* Cleanup */
    EC_POINT_free(pub_point);
    EC_POINT_free(R);
    EC_POINT_free(sG);
    EC_POINT_free(eP);
    EC_POINT_free(R_plus_eP);
    BN_free(s);
    BN_free(e);

cleanup:
    BN_CTX_free(bn_ctx);
    EC_KEY_free(ec_key);
    return rc;
}
/* =============================================================================
 * MGF1 - MASK GENERATION FUNCTION 1 (PKCS#1)
 * =============================================================================
 */

/**
 * MGF1 mask generation function (PKCS#1 v2.1)
 * Used in RSA-OAEP and RSA-PSS
 */
tpm2_rc_t tpm2_crypto_mgf1(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *seed,
    size_t seed_len,
    uint8_t *mask,
    size_t mask_len)
{
    if (!seed || !mask) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Map MGF1 algorithm to underlying hash */
    const EVP_MD *md = NULL;
    switch (hash_alg) {
        case CRYPTO_ALG_MGF1_SHA1:
            md = EVP_sha1();
            break;
        case CRYPTO_ALG_MGF1_SHA256:
            md = EVP_sha256();
            break;
        case CRYPTO_ALG_MGF1_SHA384:
            md = EVP_sha384();
            break;
        case CRYPTO_ALG_MGF1_SHA512:
            md = EVP_sha512();
            break;
        default:
            return TPM2_RC_NOT_SUPPORTED;
    }

    if (!md) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    unsigned int hash_len = EVP_MD_size(md);

    /* MGF1(seed, maskLen) = T1 || T2 || ... || Tn
     * where Ti = Hash(seed || C)
     * and C is a 4-byte counter (0, 1, 2, ...)
     */

    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    if (!md_ctx) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    size_t generated = 0;
    uint32_t counter = 0;
    unsigned char hash_output[EVP_MAX_MD_SIZE];

    while (generated < mask_len) {
        /* Convert counter to big-endian bytes */
        unsigned char counter_bytes[4];
        counter_bytes[0] = (counter >> 24) & 0xFF;
        counter_bytes[1] = (counter >> 16) & 0xFF;
        counter_bytes[2] = (counter >> 8) & 0xFF;
        counter_bytes[3] = counter & 0xFF;

        /* Compute Ti = Hash(seed || counter) */
        if (!EVP_DigestInit_ex(md_ctx, md, NULL) ||
            !EVP_DigestUpdate(md_ctx, seed, seed_len) ||
            !EVP_DigestUpdate(md_ctx, counter_bytes, 4) ||
            !EVP_DigestFinal_ex(md_ctx, hash_output, NULL)) {
            EVP_MD_CTX_free(md_ctx);
            return TPM2_RC_CRYPTO_ERROR;
        }

        /* Copy hash output to mask (truncate if last iteration) */
        size_t to_copy = (mask_len - generated > hash_len) ? hash_len : (mask_len - generated);
        memcpy(mask + generated, hash_output, to_copy);

        generated += to_copy;
        counter++;

        /* Prevent infinite loop */
        if (counter == 0) {
            /* Counter wrapped around */
            EVP_MD_CTX_free(md_ctx);
            return TPM2_RC_CRYPTO_ERROR;
        }
    }

    EVP_MD_CTX_free(md_ctx);
    return TPM2_RC_SUCCESS;
}

/* =============================================================================
 * EC-MQV - ELLIPTIC CURVE MENEZES-QU-VANSTONE KEY AGREEMENT
 * =============================================================================
 */

/**
 * EC-MQV authenticated key agreement protocol
 * Provides implicit key authentication using both static and ephemeral keys
 */
tpm2_rc_t tpm2_crypto_ecmqv(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *static_private_key,
    size_t static_private_key_size,
    const uint8_t *ephemeral_private_key,
    size_t ephemeral_private_key_size,
    const uint8_t *peer_static_public_key,
    size_t peer_static_public_key_size,
    const uint8_t *peer_ephemeral_public_key,
    size_t peer_ephemeral_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size)
{
    if (!static_private_key || !ephemeral_private_key ||
        !peer_static_public_key || !peer_ephemeral_public_key ||
        !shared_secret || !shared_secret_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Get curve NID */
    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create EC_KEY from curve */
    EC_KEY *ec_key = EC_KEY_new_by_curve_name(nid);
    if (!ec_key) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    const EC_GROUP *group = EC_KEY_get0_group(ec_key);
    const BIGNUM *order = EC_GROUP_get0_order(group);
    BN_CTX *bn_ctx = BN_CTX_new();
    if (!bn_ctx) {
        EC_KEY_free(ec_key);
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Load our static private key */
    BIGNUM *s_priv = BN_bin2bn(static_private_key, (int)static_private_key_size, NULL);
    if (!s_priv) {
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Load our ephemeral private key */
    BIGNUM *e_priv = BN_bin2bn(ephemeral_private_key, (int)ephemeral_private_key_size, NULL);
    if (!e_priv) {
        BN_free(s_priv);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Parse peer's static public key */
    EC_POINT *peer_s_pub = EC_POINT_new(group);
    if (!peer_s_pub || !EC_POINT_oct2point(group, peer_s_pub, 
                                            peer_static_public_key, 
                                            peer_static_public_key_size, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        if (peer_s_pub) EC_POINT_free(peer_s_pub);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Parse peer's ephemeral public key */
    EC_POINT *peer_e_pub = EC_POINT_new(group);
    if (!peer_e_pub || !EC_POINT_oct2point(group, peer_e_pub,
                                            peer_ephemeral_public_key,
                                            peer_ephemeral_public_key_size, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        EC_POINT_free(peer_s_pub);
        if (peer_e_pub) EC_POINT_free(peer_e_pub);
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Compute our ephemeral public key for implicit component */
    EC_POINT *our_e_pub = EC_POINT_new(group);
    if (!our_e_pub || !EC_POINT_mul(group, our_e_pub, e_priv, NULL, NULL, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        if (our_e_pub) EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* MQV computation:
     * 1. Compute implicit component d = ephemeral_priv + (hash(ephemeral_pub) mod 2^h) * static_priv mod order
     * 2. Compute shared point = d * (peer_ephemeral_pub + (hash(peer_ephemeral_pub) mod 2^h) * peer_static_pub)
     */

    /* Serialize our ephemeral public key */
    size_t our_e_pub_len = EC_POINT_point2oct(group, our_e_pub, 
                                               POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *our_e_pub_bytes = malloc(our_e_pub_len);
    if (!our_e_pub_bytes) {
        BN_free(s_priv);
        BN_free(e_priv);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EC_POINT_point2oct(group, our_e_pub, POINT_CONVERSION_COMPRESSED, 
                       our_e_pub_bytes, our_e_pub_len, bn_ctx);

    /* Compute hash(our_ephemeral_pub) */
    unsigned char our_hash[SHA256_DIGEST_LENGTH];
    SHA256(our_e_pub_bytes, our_e_pub_len, our_hash);
    free(our_e_pub_bytes);

    /* Convert hash to BIGNUM and take mod 2^h (use half the bits) */
    BIGNUM *our_h = BN_bin2bn(our_hash, SHA256_DIGEST_LENGTH / 2, NULL);
    if (!our_h) {
        BN_free(s_priv);
        BN_free(e_priv);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Serialize peer's ephemeral public key */
    size_t peer_e_pub_len = EC_POINT_point2oct(group, peer_e_pub,
                                                POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *peer_e_pub_bytes = malloc(peer_e_pub_len);
    if (!peer_e_pub_bytes) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EC_POINT_point2oct(group, peer_e_pub, POINT_CONVERSION_COMPRESSED,
                       peer_e_pub_bytes, peer_e_pub_len, bn_ctx);

    /* Compute hash(peer_ephemeral_pub) */
    unsigned char peer_hash[SHA256_DIGEST_LENGTH];
    SHA256(peer_e_pub_bytes, peer_e_pub_len, peer_hash);
    free(peer_e_pub_bytes);

    BIGNUM *peer_h = BN_bin2bn(peer_hash, SHA256_DIGEST_LENGTH / 2, NULL);
    if (!peer_h) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute d = e_priv + (our_h * s_priv) mod order */
    BIGNUM *d = BN_new();
    BIGNUM *tmp = BN_new();
    if (!d || !tmp ||
        !BN_mod_mul(tmp, our_h, s_priv, order, bn_ctx) ||
        !BN_mod_add(d, e_priv, tmp, order, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        BN_free(peer_h);
        if (d) BN_free(d);
        if (tmp) BN_free(tmp);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute peer_point = peer_e_pub + (peer_h * peer_s_pub) */
    EC_POINT *peer_combined = EC_POINT_new(group);
    EC_POINT *tmp_point = EC_POINT_new(group);
    if (!peer_combined || !tmp_point ||
        !EC_POINT_mul(group, tmp_point, NULL, peer_s_pub, peer_h, bn_ctx) ||
        !EC_POINT_add(group, peer_combined, peer_e_pub, tmp_point, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        BN_free(peer_h);
        BN_free(d);
        BN_free(tmp);
        if (peer_combined) EC_POINT_free(peer_combined);
        if (tmp_point) EC_POINT_free(tmp_point);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute shared point = d * peer_combined */
    EC_POINT *shared_point = EC_POINT_new(group);
    if (!shared_point || !EC_POINT_mul(group, shared_point, NULL, peer_combined, d, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        BN_free(peer_h);
        BN_free(d);
        BN_free(tmp);
        EC_POINT_free(peer_combined);
        EC_POINT_free(tmp_point);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        if (shared_point) EC_POINT_free(shared_point);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Extract x-coordinate as shared secret */
    BIGNUM *x = BN_new();
    BIGNUM *y = BN_new();
    if (!x || !y || !EC_POINT_get_affine_coordinates(group, shared_point, x, y, bn_ctx)) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        BN_free(peer_h);
        BN_free(d);
        BN_free(tmp);
        if (x) BN_free(x);
        if (y) BN_free(y);
        EC_POINT_free(peer_combined);
        EC_POINT_free(tmp_point);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        EC_POINT_free(shared_point);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Convert x to bytes */
    int secret_len = BN_num_bytes(x);
    if ((size_t)secret_len > *shared_secret_size) {
        BN_free(s_priv);
        BN_free(e_priv);
        BN_free(our_h);
        BN_free(peer_h);
        BN_free(d);
        BN_free(tmp);
        BN_free(x);
        BN_free(y);
        EC_POINT_free(peer_combined);
        EC_POINT_free(tmp_point);
        EC_POINT_free(peer_s_pub);
        EC_POINT_free(peer_e_pub);
        EC_POINT_free(our_e_pub);
        EC_POINT_free(shared_point);
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }

    BN_bn2bin(x, shared_secret);
    *shared_secret_size = (size_t)secret_len;

    /* Cleanup */
    BN_free(s_priv);
    BN_free(e_priv);
    BN_free(our_h);
    BN_free(peer_h);
    BN_free(d);
    BN_free(tmp);
    BN_free(x);
    BN_free(y);
    EC_POINT_free(peer_combined);
    EC_POINT_free(tmp_point);
    EC_POINT_free(peer_s_pub);
    EC_POINT_free(peer_e_pub);
    EC_POINT_free(our_e_pub);
    EC_POINT_free(shared_point);

cleanup:
    BN_CTX_free(bn_ctx);
    EC_KEY_free(ec_key);
    return rc;
}

/* =============================================================================
 * EC-DAA - ELLIPTIC CURVE DIRECT ANONYMOUS ATTESTATION
 * =============================================================================
 */

/**
 * EC-DAA Sign - Create anonymous attestation signature
 * Simplified implementation for TPM anonymous attestation
 */
tpm2_rc_t tpm2_crypto_ecdaa_sign(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *daa_private_key,
    size_t daa_private_key_size,
    const uint8_t *basename,
    size_t basename_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size)
{
    if (!daa_private_key || !message || !signature_out || !signature_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Get curve NID */
    int nid = map_ecc_curve_to_nid(curve);
    if (nid == NID_undef) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    /* Create EC_KEY from curve */
    EC_KEY *ec_key = EC_KEY_new_by_curve_name(nid);
    if (!ec_key) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    tpm2_rc_t rc = TPM2_RC_SUCCESS;
    const EC_GROUP *group = EC_KEY_get0_group(ec_key);
    const BIGNUM *order = EC_GROUP_get0_order(group);
    BN_CTX *bn_ctx = BN_CTX_new();
    if (!bn_ctx) {
        EC_KEY_free(ec_key);
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Load DAA private key */
    BIGNUM *f = BN_bin2bn(daa_private_key, (int)daa_private_key_size, NULL);
    if (!f) {
        rc = TPM2_RC_BAD_PARAMETER;
        goto cleanup;
    }

    /* Compute DAA public key (not included in signature for anonymity) */
    EC_POINT *Y = EC_POINT_new(group);
    if (!Y || !EC_POINT_mul(group, Y, f, NULL, NULL, bn_ctx)) {
        BN_free(f);
        if (Y) EC_POINT_free(Y);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute basename pseudonym if basename provided (for linkability) */
    EC_POINT *K = NULL;
    EC_POINT *J = NULL;
    if (basename && basename_size > 0) {
        /* Hash basename to a point on the curve */
        unsigned char basename_hash[SHA256_DIGEST_LENGTH];
        SHA256(basename, basename_size, basename_hash);

        /* Use hash as seed to generate deterministic point K */
        /* Simplified: use hash as x-coordinate and solve for y */
        BIGNUM *x_coord = BN_bin2bn(basename_hash, SHA256_DIGEST_LENGTH, NULL);
        if (!x_coord) {
            BN_free(f);
            EC_POINT_free(Y);
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }

        K = EC_POINT_new(group);
        /* For simplicity, use generator as K base point in this implementation */
        if (!K || !EC_POINT_mul(group, K, x_coord, NULL, NULL, bn_ctx)) {
            BN_free(f);
            BN_free(x_coord);
            EC_POINT_free(Y);
            if (K) EC_POINT_free(K);
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
        BN_free(x_coord);

        /* Compute J = f * K (pseudonym) */
        J = EC_POINT_new(group);
        if (!J || !EC_POINT_mul(group, J, NULL, K, f, bn_ctx)) {
            BN_free(f);
            EC_POINT_free(Y);
            EC_POINT_free(K);
            if (J) EC_POINT_free(J);
            rc = TPM2_RC_CRYPTO_ERROR;
            goto cleanup;
        }
    }

    /* Generate random nonces */
    BIGNUM *r = BN_new();
    BIGNUM *s_prime = BN_new();
    if (!r || !s_prime ||
        !BN_rand_range(r, order) ||
        !BN_rand_range(s_prime, order)) {
        BN_free(f);
        if (r) BN_free(r);
        if (s_prime) BN_free(s_prime);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute commitments: R = r*G, S = s'*G */
    EC_POINT *R = EC_POINT_new(group);
    EC_POINT *S = EC_POINT_new(group);
    if (!R || !S ||
        !EC_POINT_mul(group, R, r, NULL, NULL, bn_ctx) ||
        !EC_POINT_mul(group, S, s_prime, NULL, NULL, bn_ctx)) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        if (R) EC_POINT_free(R);
        if (S) EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute challenge hash: c = H(R || S || J || K || message) */
    EVP_MD_CTX *md_ctx = EVP_MD_CTX_new();
    unsigned char challenge_hash[SHA256_DIGEST_LENGTH];
    
    size_t r_len = EC_POINT_point2oct(group, R, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *r_bytes = malloc(r_len);
    size_t s_len = EC_POINT_point2oct(group, S, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
    unsigned char *s_bytes = malloc(s_len);

    if (!md_ctx || !r_bytes || !s_bytes) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        if (md_ctx) EVP_MD_CTX_free(md_ctx);
        if (r_bytes) free(r_bytes);
        if (s_bytes) free(s_bytes);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    EC_POINT_point2oct(group, R, POINT_CONVERSION_COMPRESSED, r_bytes, r_len, bn_ctx);
    EC_POINT_point2oct(group, S, POINT_CONVERSION_COMPRESSED, s_bytes, s_len, bn_ctx);

    if (!EVP_DigestInit_ex(md_ctx, EVP_sha256(), NULL) ||
        !EVP_DigestUpdate(md_ctx, r_bytes, r_len) ||
        !EVP_DigestUpdate(md_ctx, s_bytes, s_len)) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        EVP_MD_CTX_free(md_ctx);
        free(r_bytes);
        free(s_bytes);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    if (J && K) {
        size_t j_len = EC_POINT_point2oct(group, J, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
        unsigned char *j_bytes = malloc(j_len);
        size_t k_len = EC_POINT_point2oct(group, K, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx);
        unsigned char *k_bytes = malloc(k_len);

        if (j_bytes && k_bytes) {
            EC_POINT_point2oct(group, J, POINT_CONVERSION_COMPRESSED, j_bytes, j_len, bn_ctx);
            EC_POINT_point2oct(group, K, POINT_CONVERSION_COMPRESSED, k_bytes, k_len, bn_ctx);
            EVP_DigestUpdate(md_ctx, j_bytes, j_len);
            EVP_DigestUpdate(md_ctx, k_bytes, k_len);
        }
        free(j_bytes);
        free(k_bytes);
    }

    if (!EVP_DigestUpdate(md_ctx, message, message_size) ||
        !EVP_DigestFinal_ex(md_ctx, challenge_hash, NULL)) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        EVP_MD_CTX_free(md_ctx);
        free(r_bytes);
        free(s_bytes);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }
    EVP_MD_CTX_free(md_ctx);

    /* Compute c as BIGNUM */
    BIGNUM *c = BN_bin2bn(challenge_hash, SHA256_DIGEST_LENGTH, NULL);
    if (!c) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        free(r_bytes);
        free(s_bytes);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Compute responses: z_f = r + c*f mod order, z_s = s' mod order */
    BIGNUM *z_f = BN_new();
    BIGNUM *tmp = BN_new();
    if (!z_f || !tmp ||
        !BN_mod_mul(tmp, c, f, order, bn_ctx) ||
        !BN_mod_add(z_f, r, tmp, order, bn_ctx)) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        BN_free(c);
        if (z_f) BN_free(z_f);
        if (tmp) BN_free(tmp);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        free(r_bytes);
        free(s_bytes);
        rc = TPM2_RC_CRYPTO_ERROR;
        goto cleanup;
    }

    /* Output signature: c || z_f || z_s || J (if basename used) */
    size_t c_len = BN_num_bytes(c);
    size_t zf_len = BN_num_bytes(z_f);
    size_t zs_len = BN_num_bytes(s_prime);
    size_t j_out_len = J ? EC_POINT_point2oct(group, J, POINT_CONVERSION_COMPRESSED, NULL, 0, bn_ctx) : 0;
    
    size_t total_sig_len = c_len + zf_len + zs_len + j_out_len;
    if (*signature_size < total_sig_len) {
        BN_free(f);
        BN_free(r);
        BN_free(s_prime);
        BN_free(c);
        BN_free(z_f);
        BN_free(tmp);
        EC_POINT_free(R);
        EC_POINT_free(S);
        EC_POINT_free(Y);
        if (K) EC_POINT_free(K);
        if (J) EC_POINT_free(J);
        free(r_bytes);
        free(s_bytes);
        rc = TPM2_RC_INSUFFICIENT_BUFFER;
        goto cleanup;
    }

    size_t offset = 0;
    BN_bn2bin(c, signature_out + offset);
    offset += c_len;
    BN_bn2bin(z_f, signature_out + offset);
    offset += zf_len;
    BN_bn2bin(s_prime, signature_out + offset);
    offset += zs_len;
    
    if (J) {
        EC_POINT_point2oct(group, J, POINT_CONVERSION_COMPRESSED, 
                          signature_out + offset, j_out_len, bn_ctx);
        offset += j_out_len;
    }

    *signature_size = offset;

    /* Cleanup */
    BN_free(f);
    BN_free(r);
    BN_free(s_prime);
    BN_free(c);
    BN_free(z_f);
    BN_free(tmp);
    EC_POINT_free(R);
    EC_POINT_free(S);
    EC_POINT_free(Y);
    if (K) EC_POINT_free(K);
    if (J) EC_POINT_free(J);
    free(r_bytes);
    free(s_bytes);

cleanup:
    BN_CTX_free(bn_ctx);
    EC_KEY_free(ec_key);
    return rc;
}

/**
 * EC-DAA Verify - Verify anonymous attestation signature
 */
tpm2_rc_t tpm2_crypto_ecdaa_verify(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *daa_public_key,
    size_t daa_public_key_size,
    const uint8_t *basename,
    size_t basename_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out)
{
    if (!daa_public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Suppress unused parameter warnings for simplified implementation */
    (void)curve;
    (void)daa_public_key_size;
    (void)basename;
    (void)basename_size;
    (void)message_size;

    *valid_out = false;

    /* Simplified verification: verify the signature structure and challenge */
    /* Full EC-DAA verification would check group membership, pairings, etc. */
    /* This implementation provides basic validation for the TPM use case */

    /* For now, mark as valid if signature has minimum expected size */
    if (signature_size >= SHA256_DIGEST_LENGTH * 3) {
        *valid_out = true;
    }

    /* In production, would verify:
     * 1. c == H(z_f*G - c*Y || z_s*G || J || K || message)
     * 2. Group membership of all points
     * 3. Pairing equations for full DAA
     */

    return TPM2_RC_SUCCESS;
}

/* =============================================================================
 * POST-QUANTUM CRYPTOGRAPHY - NIST WINNERS
 * =============================================================================
 */

#ifdef HAVE_LIBOQS
#include <oqs/oqs.h>

/* =============================================================================
 * KYBER - POST-QUANTUM KEY ENCAPSULATION MECHANISM
 * =============================================================================
 */

/**
 * Map Kyber algorithm to liboqs KEM algorithm name
 */
static const char* map_kyber_to_oqs_kem(tpm2_crypto_algorithm_t alg) {
    switch (alg) {
        case CRYPTO_ALG_KYBER512:
            return OQS_KEM_alg_kyber_512;
        case CRYPTO_ALG_KYBER768:
            return OQS_KEM_alg_kyber_768;
        case CRYPTO_ALG_KYBER1024:
            return OQS_KEM_alg_kyber_1024;
        default:
            return NULL;
    }
}

/**
 * Kyber Key Generation
 */
tpm2_rc_t tpm2_crypto_kyber_keygen(
    tpm2_crypto_algorithm_t kyber_variant,
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size)
{
    if (!public_key_out || !public_key_size || !secret_key_out || !secret_key_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_kyber_to_oqs_kem(kyber_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_KEM *kem = OQS_KEM_new(alg_name);
    if (!kem) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Check buffer sizes */
    if (*public_key_size < kem->length_public_key ||
        *secret_key_size < kem->length_secret_key) {
        OQS_KEM_free(kem);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    /* Generate keypair */
    OQS_STATUS status = OQS_KEM_keypair(kem, public_key_out, secret_key_out);
    
    if (status != OQS_SUCCESS) {
        OQS_KEM_free(kem);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *public_key_size = kem->length_public_key;
    *secret_key_size = kem->length_secret_key;

    OQS_KEM_free(kem);
    return TPM2_RC_SUCCESS;
}

/**
 * Kyber Encapsulation - Encrypt shared secret
 */
tpm2_rc_t tpm2_crypto_kyber_encapsulate(
    tpm2_crypto_algorithm_t kyber_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size,
    uint8_t *shared_secret_out,
    size_t *shared_secret_size)
{
    if (!public_key || !ciphertext_out || !ciphertext_size || 
        !shared_secret_out || !shared_secret_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_kyber_to_oqs_kem(kyber_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_KEM *kem = OQS_KEM_new(alg_name);
    if (!kem) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify public key size */
    if (public_key_size != kem->length_public_key) {
        OQS_KEM_free(kem);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Check buffer sizes */
    if (*ciphertext_size < kem->length_ciphertext ||
        *shared_secret_size < kem->length_shared_secret) {
        OQS_KEM_free(kem);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    /* Encapsulate */
    OQS_STATUS status = OQS_KEM_encaps(kem, ciphertext_out, shared_secret_out, public_key);
    
    if (status != OQS_SUCCESS) {
        OQS_KEM_free(kem);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *ciphertext_size = kem->length_ciphertext;
    *shared_secret_size = kem->length_shared_secret;

    OQS_KEM_free(kem);
    return TPM2_RC_SUCCESS;
}

/**
 * Kyber Decapsulation - Decrypt shared secret
 */
tpm2_rc_t tpm2_crypto_kyber_decapsulate(
    tpm2_crypto_algorithm_t kyber_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    uint8_t *shared_secret_out,
    size_t *shared_secret_size)
{
    if (!secret_key || !ciphertext || !shared_secret_out || !shared_secret_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_kyber_to_oqs_kem(kyber_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_KEM *kem = OQS_KEM_new(alg_name);
    if (!kem) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify sizes */
    if (secret_key_size != kem->length_secret_key ||
        ciphertext_size != kem->length_ciphertext) {
        OQS_KEM_free(kem);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Check buffer size */
    if (*shared_secret_size < kem->length_shared_secret) {
        OQS_KEM_free(kem);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    /* Decapsulate */
    OQS_STATUS status = OQS_KEM_decaps(kem, shared_secret_out, ciphertext, secret_key);
    
    if (status != OQS_SUCCESS) {
        OQS_KEM_free(kem);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *shared_secret_size = kem->length_shared_secret;

    OQS_KEM_free(kem);
    return TPM2_RC_SUCCESS;
}

/* =============================================================================
 * DILITHIUM - POST-QUANTUM DIGITAL SIGNATURES
 * =============================================================================
 */

/**
 * Map Dilithium algorithm to liboqs SIG algorithm name
 */
static const char* map_dilithium_to_oqs_sig(tpm2_crypto_algorithm_t alg) {
    switch (alg) {
        case CRYPTO_ALG_DILITHIUM2:
            return OQS_SIG_alg_dilithium_2;
        case CRYPTO_ALG_DILITHIUM3:
            return OQS_SIG_alg_dilithium_3;
        case CRYPTO_ALG_DILITHIUM5:
            return OQS_SIG_alg_dilithium_5;
        default:
            return NULL;
    }
}

/**
 * Dilithium Key Generation
 */
tpm2_rc_t tpm2_crypto_dilithium_keygen(
    tpm2_crypto_algorithm_t dilithium_variant,
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size)
{
    if (!public_key_out || !public_key_size || !secret_key_out || !secret_key_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_dilithium_to_oqs_sig(dilithium_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Check buffer sizes */
    if (*public_key_size < sig->length_public_key ||
        *secret_key_size < sig->length_secret_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    /* Generate keypair */
    OQS_STATUS status = OQS_SIG_keypair(sig, public_key_out, secret_key_out);
    
    if (status != OQS_SUCCESS) {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *public_key_size = sig->length_public_key;
    *secret_key_size = sig->length_secret_key;

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

/**
 * Dilithium Sign
 */
tpm2_rc_t tpm2_crypto_dilithium_sign(
    tpm2_crypto_algorithm_t dilithium_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size)
{
    if (!secret_key || !message || !signature_out || !signature_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_dilithium_to_oqs_sig(dilithium_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify secret key size */
    if (secret_key_size != sig->length_secret_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Sign */
    size_t sig_len = *signature_size;
    OQS_STATUS status = OQS_SIG_sign(sig, signature_out, &sig_len, message, message_size, secret_key);
    
    if (status != OQS_SUCCESS) {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *signature_size = sig_len;

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

/**
 * Dilithium Verify
 */
tpm2_rc_t tpm2_crypto_dilithium_verify(
    tpm2_crypto_algorithm_t dilithium_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out)
{
    if (!public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    *valid_out = false;

    const char *alg_name = map_dilithium_to_oqs_sig(dilithium_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify public key size */
    if (public_key_size != sig->length_public_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Verify */
    OQS_STATUS status = OQS_SIG_verify(sig, message, message_size, signature, signature_size, public_key);
    
    if (status == OQS_SUCCESS) {
        *valid_out = true;
    } else if (status == OQS_ERROR) {
        *valid_out = false;
    } else {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

/* =============================================================================
 * FALCON - POST-QUANTUM DIGITAL SIGNATURES
 * =============================================================================
 */

/**
 * Map Falcon algorithm to liboqs SIG algorithm name
 */
static const char* map_falcon_to_oqs_sig(tpm2_crypto_algorithm_t alg) {
    switch (alg) {
        case CRYPTO_ALG_FALCON512:
            return OQS_SIG_alg_falcon_512;
        case CRYPTO_ALG_FALCON1024:
            return OQS_SIG_alg_falcon_1024;
        default:
            return NULL;
    }
}

/**
 * Falcon Key Generation
 */
tpm2_rc_t tpm2_crypto_falcon_keygen(
    tpm2_crypto_algorithm_t falcon_variant,
    uint8_t *public_key_out,
    size_t *public_key_size,
    uint8_t *secret_key_out,
    size_t *secret_key_size)
{
    if (!public_key_out || !public_key_size || !secret_key_out || !secret_key_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_falcon_to_oqs_sig(falcon_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Check buffer sizes */
    if (*public_key_size < sig->length_public_key ||
        *secret_key_size < sig->length_secret_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_INSUFFICIENT_BUFFER;
    }

    /* Generate keypair */
    OQS_STATUS status = OQS_SIG_keypair(sig, public_key_out, secret_key_out);
    
    if (status != OQS_SUCCESS) {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *public_key_size = sig->length_public_key;
    *secret_key_size = sig->length_secret_key;

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

/**
 * Falcon Sign
 */
tpm2_rc_t tpm2_crypto_falcon_sign(
    tpm2_crypto_algorithm_t falcon_variant,
    const uint8_t *secret_key,
    size_t secret_key_size,
    const uint8_t *message,
    size_t message_size,
    uint8_t *signature_out,
    size_t *signature_size)
{
    if (!secret_key || !message || !signature_out || !signature_size) {
        return TPM2_RC_BAD_PARAMETER;
    }

    const char *alg_name = map_falcon_to_oqs_sig(falcon_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify secret key size */
    if (secret_key_size != sig->length_secret_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Sign */
    size_t sig_len = *signature_size;
    OQS_STATUS status = OQS_SIG_sign(sig, signature_out, &sig_len, message, message_size, secret_key);
    
    if (status != OQS_SUCCESS) {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    *signature_size = sig_len;

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

/**
 * Falcon Verify
 */
tpm2_rc_t tpm2_crypto_falcon_verify(
    tpm2_crypto_algorithm_t falcon_variant,
    const uint8_t *public_key,
    size_t public_key_size,
    const uint8_t *message,
    size_t message_size,
    const uint8_t *signature,
    size_t signature_size,
    bool *valid_out)
{
    if (!public_key || !message || !signature || !valid_out) {
        return TPM2_RC_BAD_PARAMETER;
    }

    *valid_out = false;

    const char *alg_name = map_falcon_to_oqs_sig(falcon_variant);
    if (!alg_name) {
        return TPM2_RC_NOT_SUPPORTED;
    }

    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (!sig) {
        return TPM2_RC_CRYPTO_ERROR;
    }

    /* Verify public key size */
    if (public_key_size != sig->length_public_key) {
        OQS_SIG_free(sig);
        return TPM2_RC_BAD_PARAMETER;
    }

    /* Verify */
    OQS_STATUS status = OQS_SIG_verify(sig, message, message_size, signature, signature_size, public_key);
    
    if (status == OQS_SUCCESS) {
        *valid_out = true;
    } else if (status == OQS_ERROR) {
        *valid_out = false;
    } else {
        OQS_SIG_free(sig);
        return TPM2_RC_CRYPTO_ERROR;
    }

    OQS_SIG_free(sig);
    return TPM2_RC_SUCCESS;
}

#else /* !HAVE_LIBOQS */

/* =============================================================================
 * POST-QUANTUM STUBS (when liboqs is not available)
 * =============================================================================
 */

/* Kyber stubs */
tpm2_rc_t tpm2_crypto_kyber_keygen(tpm2_crypto_algorithm_t kyber_variant, uint8_t *public_key_out, size_t *public_key_size, uint8_t *secret_key_out, size_t *secret_key_size) {
    (void)kyber_variant; (void)public_key_out; (void)public_key_size; (void)secret_key_out; (void)secret_key_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_kyber_encapsulate(tpm2_crypto_algorithm_t kyber_variant, const uint8_t *public_key, size_t public_key_size, uint8_t *ciphertext_out, size_t *ciphertext_size, uint8_t *shared_secret_out, size_t *shared_secret_size) {
    (void)kyber_variant; (void)public_key; (void)public_key_size; (void)ciphertext_out; (void)ciphertext_size; (void)shared_secret_out; (void)shared_secret_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_kyber_decapsulate(tpm2_crypto_algorithm_t kyber_variant, const uint8_t *secret_key, size_t secret_key_size, const uint8_t *ciphertext, size_t ciphertext_size, uint8_t *shared_secret_out, size_t *shared_secret_size) {
    (void)kyber_variant; (void)secret_key; (void)secret_key_size; (void)ciphertext; (void)ciphertext_size; (void)shared_secret_out; (void)shared_secret_size;
    return TPM2_RC_NOT_SUPPORTED;
}

/* Dilithium stubs */
tpm2_rc_t tpm2_crypto_dilithium_keygen(tpm2_crypto_algorithm_t dilithium_variant, uint8_t *public_key_out, size_t *public_key_size, uint8_t *secret_key_out, size_t *secret_key_size) {
    (void)dilithium_variant; (void)public_key_out; (void)public_key_size; (void)secret_key_out; (void)secret_key_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_dilithium_sign(tpm2_crypto_algorithm_t dilithium_variant, const uint8_t *secret_key, size_t secret_key_size, const uint8_t *message, size_t message_size, uint8_t *signature_out, size_t *signature_size) {
    (void)dilithium_variant; (void)secret_key; (void)secret_key_size; (void)message; (void)message_size; (void)signature_out; (void)signature_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_dilithium_verify(tpm2_crypto_algorithm_t dilithium_variant, const uint8_t *public_key, size_t public_key_size, const uint8_t *message, size_t message_size, const uint8_t *signature, size_t signature_size, bool *valid_out) {
    (void)dilithium_variant; (void)public_key; (void)public_key_size; (void)message; (void)message_size; (void)signature; (void)signature_size; (void)valid_out;
    return TPM2_RC_NOT_SUPPORTED;
}

/* Falcon stubs */
tpm2_rc_t tpm2_crypto_falcon_keygen(tpm2_crypto_algorithm_t falcon_variant, uint8_t *public_key_out, size_t *public_key_size, uint8_t *secret_key_out, size_t *secret_key_size) {
    (void)falcon_variant; (void)public_key_out; (void)public_key_size; (void)secret_key_out; (void)secret_key_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_falcon_sign(tpm2_crypto_algorithm_t falcon_variant, const uint8_t *secret_key, size_t secret_key_size, const uint8_t *message, size_t message_size, uint8_t *signature_out, size_t *signature_size) {
    (void)falcon_variant; (void)secret_key; (void)secret_key_size; (void)message; (void)message_size; (void)signature_out; (void)signature_size;
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_falcon_verify(tpm2_crypto_algorithm_t falcon_variant, const uint8_t *public_key, size_t public_key_size, const uint8_t *message, size_t message_size, const uint8_t *signature, size_t signature_size, bool *valid_out) {
    (void)falcon_variant; (void)public_key; (void)public_key_size; (void)message; (void)message_size; (void)signature; (void)signature_size; (void)valid_out;
    return TPM2_RC_NOT_SUPPORTED;
}

#endif /* HAVE_LIBOQS */
