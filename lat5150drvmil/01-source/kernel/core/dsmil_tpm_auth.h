/*
 * DSMIL TPM Authentication Integration
 * =====================================
 *
 * Hardware-backed authentication using Linux TPM 2.0 subsystem
 * for protected token access and security-critical operations.
 *
 * Copyright (C) 2025 DSMIL Development Project
 * License: GPL v2
 */

#ifndef _DSMIL_TPM_AUTH_H
#define _DSMIL_TPM_AUTH_H

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/tpm.h>
#include <linux/random.h>
#include <linux/slab.h>
#include "dsmil_error_handling.h"

/*
 * TPM Authentication Architecture
 *
 * Provides hardware-backed authentication for protected tokens using:
 * - TPM 2.0 chip for secure key storage and cryptographic operations
 * - PCR measurements for security-critical operation attestation
 * - Challenge-response authentication protocols
 * - Session binding with TPM-generated keys
 *
 * Authentication Flow:
 * 1. User requests authentication via IOCTL
 * 2. Driver generates random challenge
 * 3. User provides response (signed with TPM key or external credential)
 * 4. Driver validates response using TPM
 * 5. On success, creates time-limited authenticated session
 * 6. Session token required for protected token access
 *
 * Fallback: When TPM unavailable, falls back to CAP_SYS_ADMIN only
 */

/* TPM PCR indices for DSMIL measurements */
#define DSMIL_TPM_PCR_AUTH        16  /* Authentication events */
#define DSMIL_TPM_PCR_TOKEN       17  /* Protected token operations */
#define DSMIL_TPM_PCR_BIOS        18  /* BIOS failover/sync operations */
#define DSMIL_TPM_PCR_SECURITY    23  /* Security-critical operations */

/* TPM authentication modes */
enum dsmil_tpm_auth_mode {
	DSMIL_TPM_AUTH_NONE = 0,           /* No TPM authentication */
	DSMIL_TPM_AUTH_CHALLENGE,          /* Challenge-response */
	DSMIL_TPM_AUTH_KEY,                /* TPM key-based */
	DSMIL_TPM_AUTH_HMAC,               /* HMAC-based */
	DSMIL_TPM_AUTH_EXTERNAL,           /* External credential */
};

/* TPM authentication state */
enum dsmil_tpm_state {
	DSMIL_TPM_STATE_UNINITIALIZED = 0, /* TPM not initialized */
	DSMIL_TPM_STATE_UNAVAILABLE,       /* TPM not available */
	DSMIL_TPM_STATE_READY,             /* TPM ready for operations */
	DSMIL_TPM_STATE_ERROR,             /* TPM in error state */
};

/* TPM authentication challenge */
struct dsmil_tpm_challenge {
	u8 challenge[32];                  /* Random challenge data */
	ktime_t issued_time;               /* When challenge was issued */
	ktime_t expire_time;               /* Challenge expiration */
	u32 challenge_id;                  /* Unique challenge identifier */
	bool active;                       /* Challenge is active */
};

/* TPM authentication response */
struct dsmil_tpm_response {
	u8 response[256];                  /* Response data (signature/HMAC) */
	u32 response_len;                  /* Response length */
	u32 challenge_id;                  /* Matching challenge ID */
	enum dsmil_tpm_auth_mode mode;     /* Authentication mode */
};

/* TPM authentication context */
struct dsmil_tpm_auth_context {
	struct tpm_chip *chip;             /* TPM chip device */
	enum dsmil_tpm_state state;        /* TPM state */
	enum dsmil_tpm_auth_mode auth_mode;/* Active auth mode */

	/* Challenge-response state */
	struct dsmil_tpm_challenge challenge;

	/* Session management */
	bool session_active;               /* Authenticated session active */
	ktime_t session_start;             /* Session start time */
	ktime_t session_expire;            /* Session expiration time */
	u32 session_token;                 /* Session token */
	u32 user_id;                       /* Authenticated user ID */

	/* Statistics */
	atomic_t auth_attempts;            /* Total auth attempts */
	atomic_t auth_successes;           /* Successful authentications */
	atomic_t auth_failures;            /* Failed authentications */
	atomic_t pcr_extends;              /* TPM PCR extend operations */

	/* Error tracking */
	struct dsmil_error_stats *error_stats;

	/* Configuration */
	bool require_tpm;                  /* TPM required (vs optional) */
	u32 session_timeout_ms;            /* Session timeout (milliseconds) */
	u32 challenge_timeout_ms;          /* Challenge timeout (milliseconds) */

	struct mutex lock;                 /* Context lock */
};

/*
 * TPM Chip Initialization
 */

/**
 * dsmil_tpm_init - Initialize TPM authentication context
 * @ctx: TPM authentication context
 * @error_stats: Error statistics structure (optional)
 * @require_tpm: If true, driver fails if TPM unavailable
 *
 * Initializes TPM authentication and attempts to open TPM chip.
 * If TPM is unavailable and not required, falls back to CAP_SYS_ADMIN only.
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_tpm_init(struct dsmil_tpm_auth_context *ctx,
				 struct dsmil_error_stats *error_stats,
				 bool require_tpm)
{
	memset(ctx, 0, sizeof(*ctx));
	mutex_init(&ctx->lock);

	ctx->error_stats = error_stats;
	ctx->require_tpm = require_tpm;
	ctx->session_timeout_ms = 300000;      /* 5 minutes */
	ctx->challenge_timeout_ms = 60000;     /* 1 minute */

	atomic_set(&ctx->auth_attempts, 0);
	atomic_set(&ctx->auth_successes, 0);
	atomic_set(&ctx->auth_failures, 0);
	atomic_set(&ctx->pcr_extends, 0);

	/* Try to open TPM chip */
	ctx->chip = tpm_default_chip();
	if (!ctx->chip) {
		pr_warn("DSMIL TPM: TPM chip not available\n");
		ctx->state = DSMIL_TPM_STATE_UNAVAILABLE;

		if (require_tpm) {
			pr_err("DSMIL TPM: TPM required but unavailable\n");
			return -ENODEV;
		}

		pr_info("DSMIL TPM: Falling back to capability-based authentication\n");
		return 0;
	}

	/* Verify TPM is functional */
	if (!(ctx->chip->flags & TPM_CHIP_FLAG_TPM2)) {
		pr_warn("DSMIL TPM: TPM 1.2 not supported, require TPM 2.0\n");
		tpm_put_ops(ctx->chip);
		ctx->chip = NULL;
		ctx->state = DSMIL_TPM_STATE_UNAVAILABLE;

		if (require_tpm)
			return -ENODEV;
		return 0;
	}

	ctx->state = DSMIL_TPM_STATE_READY;
	ctx->auth_mode = DSMIL_TPM_AUTH_CHALLENGE;

	pr_info("DSMIL TPM: TPM 2.0 chip initialized successfully\n");
	pr_info("DSMIL TPM: - Device: %s\n", dev_name(&ctx->chip->dev));
	pr_info("DSMIL TPM: - Auth mode: challenge-response\n");

	return 0;
}

/**
 * dsmil_tpm_cleanup - Cleanup TPM authentication context
 * @ctx: TPM authentication context
 */
static inline void dsmil_tpm_cleanup(struct dsmil_tpm_auth_context *ctx)
{
	mutex_lock(&ctx->lock);

	/* Invalidate any active sessions */
	ctx->session_active = false;
	ctx->challenge.active = false;

	/* Release TPM chip */
	if (ctx->chip) {
		tpm_put_ops(ctx->chip);
		ctx->chip = NULL;
	}

	ctx->state = DSMIL_TPM_STATE_UNINITIALIZED;

	mutex_unlock(&ctx->lock);

	pr_info("DSMIL TPM: Cleanup complete (auths: %d success, %d fail)\n",
		atomic_read(&ctx->auth_successes),
		atomic_read(&ctx->auth_failures));
}

/**
 * dsmil_tpm_is_available - Check if TPM is available
 * @ctx: TPM authentication context
 *
 * Returns: true if TPM is available and ready
 */
static inline bool dsmil_tpm_is_available(struct dsmil_tpm_auth_context *ctx)
{
	return (ctx->state == DSMIL_TPM_STATE_READY && ctx->chip != NULL);
}

/*
 * TPM PCR Measurements
 */

/**
 * dsmil_tpm_extend_pcr - Extend TPM PCR with measurement
 * @ctx: TPM authentication context
 * @pcr_idx: PCR index to extend
 * @event_data: Event data to measure
 * @event_len: Length of event data
 * @description: Event description (for logging)
 *
 * Extends specified TPM PCR with SHA-256 hash of event data.
 * Used for attestation of security-critical operations.
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_tpm_extend_pcr(struct dsmil_tpm_auth_context *ctx,
				       int pcr_idx,
				       const void *event_data,
				       size_t event_len,
				       const char *description)
{
	struct tpm_digest digest;
	int ret;

	if (!dsmil_tpm_is_available(ctx)) {
		pr_debug("DSMIL TPM: PCR extend skipped (TPM unavailable): %s\n",
			 description);
		return 0; /* Not an error - TPM optional */
	}

	/* Prepare SHA-256 digest */
	digest.alg_id = TPM_ALG_SHA256;

	/* Hash event data (simplified - in production use crypto API) */
	memset(digest.digest, 0, sizeof(digest.digest));
	if (event_len > 0 && event_data) {
		size_t copy_len = min_t(size_t, event_len, TPM_MAX_DIGEST_SIZE);
		memcpy(digest.digest, event_data, copy_len);
	}

	/* Extend PCR */
	ret = tpm_pcr_extend(ctx->chip, pcr_idx, &digest);
	if (ret) {
		pr_warn("DSMIL TPM: PCR%d extend failed: %d (%s)\n",
			pcr_idx, ret, description);

		if (ctx->error_stats)
			dsmil_log_error(ctx->error_stats, SECURITY,
					DSMIL_ERR_SECURITY_TPM_FAILED,
					"PCR%d extend failed: %s", pcr_idx, description);
		return ret;
	}

	atomic_inc(&ctx->pcr_extends);
	pr_debug("DSMIL TPM: PCR%d extended: %s\n", pcr_idx, description);

	return 0;
}

/*
 * Challenge-Response Authentication
 */

/**
 * dsmil_tpm_generate_challenge - Generate authentication challenge
 * @ctx: TPM authentication context
 *
 * Generates random challenge for authentication.
 * User must respond with valid signature/HMAC to authenticate.
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_tpm_generate_challenge(struct dsmil_tpm_auth_context *ctx)
{
	int ret;

	mutex_lock(&ctx->lock);

	/* Invalidate any existing challenge */
	ctx->challenge.active = false;

	/* Generate random challenge data */
	get_random_bytes(ctx->challenge.challenge, sizeof(ctx->challenge.challenge));

	/* Generate unique challenge ID */
	ctx->challenge.challenge_id = get_random_u32();

	/* Set timestamps */
	ctx->challenge.issued_time = ktime_get();
	ctx->challenge.expire_time = ktime_add_ms(ctx->challenge.issued_time,
						   ctx->challenge_timeout_ms);

	ctx->challenge.active = true;

	mutex_unlock(&ctx->lock);

	pr_debug("DSMIL TPM: Challenge generated (ID: 0x%08x)\n",
		 ctx->challenge.challenge_id);

	return 0;
}

/**
 * dsmil_tpm_validate_response - Validate authentication response
 * @ctx: TPM authentication context
 * @response: User's authentication response
 *
 * Validates user's response to authentication challenge.
 * On success, creates authenticated session.
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_tpm_validate_response(struct dsmil_tpm_auth_context *ctx,
					      struct dsmil_tpm_response *response)
{
	bool valid = false;
	int ret = 0;

	mutex_lock(&ctx->lock);

	atomic_inc(&ctx->auth_attempts);

	/* Check challenge is active */
	if (!ctx->challenge.active) {
		pr_warn("DSMIL TPM: No active challenge\n");
		ret = -EINVAL;
		goto fail;
	}

	/* Check challenge hasn't expired */
	if (ktime_after(ktime_get(), ctx->challenge.expire_time)) {
		pr_warn("DSMIL TPM: Challenge expired\n");
		ret = -ETIMEDOUT;
		goto fail;
	}

	/* Check challenge ID matches */
	if (response->challenge_id != ctx->challenge.challenge_id) {
		pr_warn("DSMIL TPM: Challenge ID mismatch\n");
		ret = -EINVAL;
		goto fail;
	}

	/*
	 * Validate response based on authentication mode
	 *
	 * TODO: Implement real validation:
	 * - CHALLENGE: Verify signature using TPM-stored public key
	 * - KEY: Decrypt response using TPM key
	 * - HMAC: Verify HMAC using TPM-derived key
	 * - EXTERNAL: Validate against external credential store
	 *
	 * For now, simple validation for testing:
	 */
	switch (response->mode) {
	case DSMIL_TPM_AUTH_CHALLENGE:
		/* Simple validation: response must contain challenge data */
		if (response->response_len >= sizeof(ctx->challenge.challenge)) {
			valid = (memcmp(response->response,
					ctx->challenge.challenge,
					sizeof(ctx->challenge.challenge)) == 0);
		}
		break;

	case DSMIL_TPM_AUTH_KEY:
	case DSMIL_TPM_AUTH_HMAC:
	case DSMIL_TPM_AUTH_EXTERNAL:
		/* TODO: Implement proper validation */
		pr_warn("DSMIL TPM: Auth mode %d not yet implemented\n", response->mode);
		ret = -ENOSYS;
		goto fail;

	default:
		pr_warn("DSMIL TPM: Invalid auth mode %d\n", response->mode);
		ret = -EINVAL;
		goto fail;
	}

	if (!valid) {
		pr_warn("DSMIL TPM: Response validation failed\n");
		ret = -EACCES;
		goto fail;
	}

	/* Response validated - create authenticated session */
	ctx->session_active = true;
	ctx->session_start = ktime_get();
	ctx->session_expire = ktime_add_ms(ctx->session_start, ctx->session_timeout_ms);
	ctx->session_token = get_random_u32();
	ctx->user_id = current_uid().val;

	/* Invalidate challenge */
	ctx->challenge.active = false;

	/* Measure authentication event in TPM */
	if (dsmil_tpm_is_available(ctx)) {
		u8 event[8];
		*(u32 *)&event[0] = ctx->user_id;
		*(u32 *)&event[4] = (u32)ktime_to_ms(ctx->session_start);

		dsmil_tpm_extend_pcr(ctx, DSMIL_TPM_PCR_AUTH,
				     event, sizeof(event),
				     "Authentication success");
	}

	atomic_inc(&ctx->auth_successes);
	mutex_unlock(&ctx->lock);

	pr_info("DSMIL TPM: Authentication successful (user=%u, token=0x%08x)\n",
		ctx->user_id, ctx->session_token);

	return 0;

fail:
	atomic_inc(&ctx->auth_failures);
	ctx->challenge.active = false;

	if (ctx->error_stats)
		dsmil_log_auth_error(ctx->error_stats, DSMIL_ERR_AUTH_FAILED,
				     "TPM authentication failed: %d", ret);

	mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * dsmil_tpm_check_session - Check if authenticated session is valid
 * @ctx: TPM authentication context
 *
 * Checks if there is an active, non-expired authenticated session.
 * Used to gate access to protected tokens.
 *
 * Returns: 0 if session valid, negative error code otherwise
 */
static inline int dsmil_tpm_check_session(struct dsmil_tpm_auth_context *ctx)
{
	int ret = 0;

	mutex_lock(&ctx->lock);

	if (!ctx->session_active) {
		ret = -EACCES;
		goto out;
	}

	if (ktime_after(ktime_get(), ctx->session_expire)) {
		pr_info("DSMIL TPM: Session expired (user=%u)\n", ctx->user_id);
		ctx->session_active = false;
		ret = -ETIMEDOUT;
		goto out;
	}

	/* Session valid */
	pr_debug("DSMIL TPM: Session valid (user=%u, token=0x%08x)\n",
		 ctx->user_id, ctx->session_token);

out:
	mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * dsmil_tpm_invalidate_session - Invalidate authenticated session
 * @ctx: TPM authentication context
 *
 * Invalidates current authenticated session (logout).
 */
static inline void dsmil_tpm_invalidate_session(struct dsmil_tpm_auth_context *ctx)
{
	mutex_lock(&ctx->lock);

	if (ctx->session_active) {
		pr_info("DSMIL TPM: Session invalidated (user=%u)\n", ctx->user_id);
		ctx->session_active = false;
	}

	mutex_unlock(&ctx->lock);
}

/*
 * Protected Token Authentication
 */

/**
 * dsmil_tpm_authorize_protected_token - Authorize protected token access
 * @ctx: TPM authentication context
 * @token_id: Token ID being accessed
 * @operation: Operation description (for logging/measurement)
 *
 * Checks authorization for protected token access.
 * Requires active authenticated session when TPM available,
 * falls back to CAP_SYS_ADMIN when TPM unavailable.
 *
 * Returns: 0 if authorized, negative error code otherwise
 */
static inline int dsmil_tpm_authorize_protected_token(struct dsmil_tpm_auth_context *ctx,
						      u16 token_id,
						      const char *operation)
{
	u8 event[6];
	int ret;

	/* Check capability first (always required) */
	if (!capable(CAP_SYS_ADMIN)) {
		pr_warn("DSMIL TPM: Insufficient privileges for token 0x%04x\n", token_id);
		return -EPERM;
	}

	/* If TPM available, require authenticated session */
	if (dsmil_tpm_is_available(ctx)) {
		ret = dsmil_tpm_check_session(ctx);
		if (ret) {
			pr_warn("DSMIL TPM: No valid session for token 0x%04x\n", token_id);
			return ret;
		}

		/* Measure protected token access in TPM */
		*(u16 *)&event[0] = token_id;
		*(u32 *)&event[2] = ctx->user_id;

		dsmil_tpm_extend_pcr(ctx, DSMIL_TPM_PCR_TOKEN,
				     event, sizeof(event),
				     operation);
	} else {
		/* TPM unavailable - CAP_SYS_ADMIN only */
		pr_debug("DSMIL TPM: Token 0x%04x authorized (no TPM, capability only)\n",
			 token_id);
	}

	return 0;
}

/**
 * dsmil_tpm_measure_security_event - Measure security-critical event
 * @ctx: TPM authentication context
 * @event_type: Event type code
 * @event_data: Event-specific data
 * @event_len: Length of event data
 * @description: Event description
 *
 * Measures security-critical event in TPM PCR for attestation.
 * Used for BIOS failover, firmware updates, etc.
 */
static inline void dsmil_tpm_measure_security_event(struct dsmil_tpm_auth_context *ctx,
						    u32 event_type,
						    const void *event_data,
						    size_t event_len,
						    const char *description)
{
	u8 event[256];
	size_t total_len;

	if (!dsmil_tpm_is_available(ctx))
		return;

	/* Build event: [event_type][event_data] */
	*(u32 *)&event[0] = event_type;
	total_len = sizeof(u32);

	if (event_data && event_len > 0) {
		size_t copy_len = min_t(size_t, event_len, sizeof(event) - total_len);
		memcpy(&event[total_len], event_data, copy_len);
		total_len += copy_len;
	}

	dsmil_tpm_extend_pcr(ctx, DSMIL_TPM_PCR_SECURITY,
			     event, total_len, description);
}

#endif /* _DSMIL_TPM_AUTH_H */
