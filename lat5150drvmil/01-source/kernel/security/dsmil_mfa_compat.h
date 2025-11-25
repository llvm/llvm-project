/*
 * DSMIL MFA Kernel Compatibility Layer
 * ====================================
 *
 * Purpose:
 *   - Hide kernel version churn behind small, well-named helpers.
 *   - Provide a stable time API and crypto cipher abstraction for DSMIL MFA.
 *
 * Notes:
 *   - Time: wraps ktime_get_real_ts64() into a simple value-returning helper.
 *   - Crypto: introduces dsmil_cipher_t as an opaque handle that maps to
 *             crypto_cipher (pre-6.4) or crypto_sync_skcipher (6.4+).
 */

#ifndef _DSMIL_MFA_COMPAT_H
#define _DSMIL_MFA_COMPAT_H

#include <linux/version.h>
#include <linux/time.h>
#include <linux/crypto.h>
#include <crypto/hash.h>

/* SHA256 digest size (32 bytes) */
#ifndef SHA256_DIGEST_SIZE
#define SHA256_DIGEST_SIZE 32
#endif

/*
 * Unified real-time accessor
 * --------------------------
 * Always use dsmil_get_real_time() in DSMIL code instead of calling
 * ktime_get_real_ts64() directly. This keeps call sites simple and
 * isolates any future signature changes here.
 */
static inline struct timespec64 dsmil_get_real_time(void)
{
	struct timespec64 ts;

	ktime_get_real_ts64(&ts);
	return ts;
}

/*
 * Symmetric cipher compatibility layer
 * ====================================
 *
 * Goal:
 *   Provide a single opaque cipher handle and alloc/free helpers that
 *   insulate DSMIL code from the crypto_cipher deprecation in >= 6.4.
 *
 * Usage in DSMIL code:
 *
 *   dsmil_cipher_t *c;
 *
 *   c = dsmil_crypto_alloc_cipher_compat("ecb(aes)");
 *   if (IS_ERR(c))
 *       return PTR_ERR(c);
 *
 *   ... use version-specific helpers (see below) ...
 *
 *   dsmil_crypto_free_cipher_compat(c);
 *
 * Implementation:
 *   - < 6.4: dsmil_cipher_t == struct crypto_cipher
 *   - >=6.4: dsmil_cipher_t == struct crypto_sync_skcipher
 *
 * Only allocation/free are abstracted here; call sites should use
 * version-aware helpers/macros for setkey/encrypt/decrypt, or keep
 * those localized behind their own tiny wrappers to avoid spray of
 * #ifdefs.
 */

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)

#include <crypto/skcipher.h>

/* Opaque DSMIL cipher handle for kernels using the skcipher API */
typedef struct crypto_sync_skcipher dsmil_cipher_t;

/**
 * dsmil_crypto_alloc_cipher_compat() - allocate DSMIL cipher handle
 * @alg_name: algorithm name, e.g. "ecb(aes)" or "cbc(aes)"
 *
 * Returns:
 *   dsmil_cipher_t * on success, or ERR_PTR(-errno) on failure.
 */
static inline dsmil_cipher_t *
dsmil_crypto_alloc_cipher_compat(const char *alg_name)
{
	return crypto_alloc_sync_skcipher(alg_name, 0, 0);
}

/**
 * dsmil_crypto_free_cipher_compat() - free DSMIL cipher handle
 */
static inline void
dsmil_crypto_free_cipher_compat(dsmil_cipher_t *tfm)
{
	if (tfm)
		crypto_free_sync_skcipher(tfm);
}

#else /* LINUX_VERSION_CODE < 6.4.0 */

#include <crypto/cipher.h>

/* Opaque DSMIL cipher handle for legacy cipher API */
typedef struct crypto_cipher dsmil_cipher_t;

static inline dsmil_cipher_t *
dsmil_crypto_alloc_cipher_compat(const char *alg_name)
{
	return crypto_alloc_cipher(alg_name, 0, 0);
}

static inline void
dsmil_crypto_free_cipher_compat(dsmil_cipher_t *tfm)
{
	if (tfm)
		crypto_free_cipher(tfm);
}

#endif /* LINUX_VERSION_CODE >= 6.4.0 */

#endif /* _DSMIL_MFA_COMPAT_H */
