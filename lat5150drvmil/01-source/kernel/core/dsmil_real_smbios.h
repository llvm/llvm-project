/*
 * DSMIL Real SMBIOS Integration
 * ==============================
 *
 * Integration with Linux kernel dell-smbios subsystem for real hardware access.
 * Falls back to simulation on non-Dell platforms or when dell-smbios is unavailable.
 *
 * Copyright (C) 2025 DSMIL Development Project
 * License: GPL v2
 */

#ifndef _DSMIL_REAL_SMBIOS_H
#define _DSMIL_REAL_SMBIOS_H

#include <linux/types.h>
#include <linux/kernel.h>
#include "dsmil_dell_smbios.h"
#include "dsmil_token_database.h"
#include "dsmil_error_handling.h"

/*
 * SMBIOS Backend Selection
 *
 * The driver supports multiple SMBIOS backends:
 * 1. Real dell-smbios (CONFIG_DELL_SMBIOS=y) - Production use on Dell hardware
 * 2. Simulated backend (fallback) - Development, testing, non-Dell platforms
 *
 * Backend is selected at compile-time and runtime based on:
 * - Kernel configuration (CONFIG_DELL_SMBIOS)
 * - Platform detection (DMI match for Dell systems)
 * - dell-smbios module availability
 */

/* Backend type enumeration */
enum dsmil_smbios_backend {
	DSMIL_BACKEND_NONE = 0,      /* No backend initialized */
	DSMIL_BACKEND_SIMULATED,     /* Simulated SMBIOS (fallback) */
	DSMIL_BACKEND_DELL_SMBIOS,   /* Real dell-smbios kernel subsystem */
};

/* Backend capabilities */
struct dsmil_smbios_backend_info {
	enum dsmil_smbios_backend backend_type;
	const char *backend_name;
	bool supports_token_read;
	bool supports_token_write;
	bool supports_token_discovery;
	bool supports_wmi;
	bool supports_smm;
	u32 max_buffer_size;
};

/* SMBIOS backend context */
struct dsmil_smbios_context {
	enum dsmil_smbios_backend backend;
	struct dsmil_smbios_backend_info info;
	struct dsmil_error_stats *error_stats;
	void *backend_priv;  /* Backend-specific private data */
	spinlock_t lock;
};

/*
 * Real dell-smbios Integration
 *
 * Uses kernel dell-smbios subsystem (drivers/platform/x86/dell/dell-smbios-base.c)
 *
 * Key Functions:
 * - dell_smbios_get_buffer() - Allocate SMBIOS buffer
 * - dell_smbios_call() - Execute SMBIOS call
 * - dell_smbios_release_buffer() - Release SMBIOS buffer
 *
 * Integration Method:
 * - Platform driver registered with dell-smbios subsystem
 * - Uses dell_smbios_register_device() to register
 * - Receives SMBIOS call results via calling_interface_buffer
 */

#ifdef CONFIG_DELL_SMBIOS
/* Real dell-smbios available - use kernel subsystem */

/* External dell-smbios functions (from drivers/platform/x86/dell/dell-smbios-base.c) */
extern struct calling_interface_buffer *dell_smbios_get_buffer(void);
extern void dell_smbios_release_buffer(void);
extern int dell_smbios_call(struct calling_interface_buffer *buffer);

/**
 * dsmil_real_smbios_call - Execute SMBIOS call via kernel dell-smbios subsystem
 * @ctx: SMBIOS backend context
 * @buffer: SMBIOS call buffer
 *
 * Returns: SMBIOS return code (SMBIOS_RET_SUCCESS, etc.)
 */
static inline int dsmil_real_smbios_call(struct dsmil_smbios_context *ctx,
					 struct calling_interface_buffer *local_buffer)
{
	struct calling_interface_buffer *buffer;
	unsigned long flags;
	int ret;

	/* Get SMBIOS buffer from dell-smbios subsystem */
	buffer = dell_smbios_get_buffer();
	if (!buffer) {
		if (ctx->error_stats)
			dsmil_log_error(ctx->error_stats, SMBIOS,
					DSMIL_ERR_SMBIOS_CALL,
					"Failed to allocate SMBIOS buffer");
		return SMBIOS_RET_UNSUPPORTED_FUNC;
	}

	/* Copy local buffer to SMBIOS buffer */
	spin_lock_irqsave(&ctx->lock, flags);
	memcpy(buffer, local_buffer, sizeof(*buffer));
	spin_unlock_irqrestore(&ctx->lock, flags);

	/* Execute SMBIOS call */
	ret = dell_smbios_call(buffer);

	/* Copy result back to local buffer */
	spin_lock_irqsave(&ctx->lock, flags);
	memcpy(local_buffer, buffer, sizeof(*local_buffer));
	spin_unlock_irqrestore(&ctx->lock, flags);

	/* Release SMBIOS buffer */
	dell_smbios_release_buffer();

	if (ret != 0) {
		if (ctx->error_stats)
			dsmil_log_error(ctx->error_stats, SMBIOS,
					DSMIL_ERR_SMBIOS_CALL,
					"SMBIOS call failed: ret=%d", ret);
		return SMBIOS_RET_UNSUPPORTED_FUNC;
	}

	return SMBIOS_RET_SUCCESS;
}

/**
 * dsmil_real_smbios_init - Initialize real dell-smbios backend
 * @ctx: SMBIOS backend context
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_real_smbios_init(struct dsmil_smbios_context *ctx)
{
	/* Check if dell-smbios is available */
	if (!dell_smbios_get_buffer) {
		pr_warn("DSMIL: dell-smbios subsystem not available\n");
		return -ENODEV;
	}

	ctx->backend = DSMIL_BACKEND_DELL_SMBIOS;
	ctx->info.backend_type = DSMIL_BACKEND_DELL_SMBIOS;
	ctx->info.backend_name = "dell-smbios (kernel subsystem)";
	ctx->info.supports_token_read = true;
	ctx->info.supports_token_write = true;
	ctx->info.supports_token_discovery = true;
	ctx->info.supports_wmi = true;
	ctx->info.supports_smm = true;
	ctx->info.max_buffer_size = sizeof(struct calling_interface_buffer);

	pr_info("DSMIL: Initialized real dell-smbios backend\n");
	return 0;
}

#else
/* CONFIG_DELL_SMBIOS not available - stubs */

static inline int dsmil_real_smbios_call(struct dsmil_smbios_context *ctx,
					 struct calling_interface_buffer *buffer)
{
	pr_debug("DSMIL: Real dell-smbios not available (CONFIG_DELL_SMBIOS=n)\n");
	return -ENOSYS;
}

static inline int dsmil_real_smbios_init(struct dsmil_smbios_context *ctx)
{
	pr_info("DSMIL: dell-smbios support not compiled in\n");
	return -ENODEV;
}

#endif /* CONFIG_DELL_SMBIOS */

/*
 * Simulated SMBIOS Backend
 *
 * Provides realistic simulation of Dell SMBIOS for:
 * - Development and testing
 * - Non-Dell platforms
 * - Systems without dell-smbios support
 *
 * Simulation includes:
 * - Database-aware token responses
 * - Realistic device status values
 * - BIOS health score simulation
 * - Token discovery support
 */

/**
 * dsmil_simulated_smbios_call - Execute simulated SMBIOS call
 * @ctx: SMBIOS backend context
 * @buffer: SMBIOS call buffer
 *
 * Returns: SMBIOS return code (SMBIOS_RET_SUCCESS, etc.)
 */
static inline int dsmil_simulated_smbios_call(struct dsmil_smbios_context *ctx,
					      struct calling_interface_buffer *buffer)
{
	u16 token_id;
	const struct dsmil_token_info *info;

	pr_debug("DSMIL: Simulated SMBIOS call class=%u select=%u input[0]=0x%08x\n",
		 buffer->cmd_class, buffer->cmd_select, buffer->input[0]);

	switch (buffer->cmd_class) {
	case CLASS_TOKEN_READ:
		token_id = buffer->input[0];
		info = dsmil_token_db_find(token_id);

		if (!info) {
			pr_debug("DSMIL: Unknown token 0x%04x\n", token_id);
			return SMBIOS_RET_INVALID_TOKEN;
		}

		/* Simulate token read based on type and token ID */
		if (token_id >= TOKEN_DSMIL_DEVICE_BASE && token_id <= 0x80FF) {
			/* Device token - simulate device status */
			u16 device_id = (token_id - TOKEN_DSMIL_DEVICE_BASE) / 3;
			u16 offset = (token_id - TOKEN_DSMIL_DEVICE_BASE) % 3;

			if (offset == TOKEN_OFFSET_STATUS) {
				/* Simulate device status: bit 0 = online, bit 1 = ready */
				buffer->output[0] = 0x00000003;
			} else if (offset == TOKEN_OFFSET_CONFIG) {
				/* Simulate device config */
				buffer->output[0] = 0x00000001;
			} else {
				/* Simulate device data */
				buffer->output[0] = 0x00000000;
			}
		} else if (token_id >= TOKEN_BIOS_A_STATUS && token_id <= TOKEN_BIOS_C_CONTROL) {
			/* BIOS token - simulate BIOS health */
			if ((token_id & 0x0F) == 0x06) {
				/* Health score token */
				u8 bios_idx = (token_id - TOKEN_BIOS_A_STATUS) / 0x10;
				/* Simulate health scores: A=90, B=85, C=95 */
				u8 health_scores[] = {90, 85, 95};
				buffer->output[0] = health_scores[bios_idx % 3];
			} else {
				/* Other BIOS tokens */
				buffer->output[0] = 0x00000000;
			}
		} else {
			/* Other tokens - return simulated value based on type */
			switch (info->token_type) {
			case DSMIL_TOKEN_TYPE_BOOL:
				buffer->output[0] = 0;
				break;
			case DSMIL_TOKEN_TYPE_U8:
			case DSMIL_TOKEN_TYPE_U16:
			case DSMIL_TOKEN_TYPE_U32:
				buffer->output[0] = (u32)info->min_value;
				break;
			default:
				buffer->output[0] = 0;
				break;
			}
		}
		break;

	case CLASS_TOKEN_WRITE:
		token_id = buffer->input[0];
		info = dsmil_token_db_find(token_id);

		if (!info) {
			pr_debug("DSMIL: Unknown token 0x%04x\n", token_id);
			return SMBIOS_RET_INVALID_TOKEN;
		}

		/* Check if read-only */
		if (info->token_flags & DSMIL_TOKEN_FLAG_READONLY) {
			pr_warn("DSMIL: Attempt to write read-only token 0x%04x\n", token_id);
			return SMBIOS_RET_PERMISSION_DENIED;
		}

		/* Simulate successful write */
		buffer->output[0] = 0;
		pr_debug("DSMIL: Token 0x%04x write simulated (value=0x%08x)\n",
			 token_id, buffer->input[2]);
		break;

	case CLASS_INFO:
		/* Token discovery/info request */
		if (buffer->input[0] >= 0x8000 && buffer->input[1] <= 0x8FFF) {
			/* Return count of tokens in DSMIL range */
			buffer->output[0] = DSMIL_TOKEN_DATABASE_SIZE;
		} else {
			buffer->output[0] = 0;
		}
		break;

	default:
		pr_warn("DSMIL: Unsupported SMBIOS class %u\n", buffer->cmd_class);
		return SMBIOS_RET_UNSUPPORTED_FUNC;
	}

	return SMBIOS_RET_SUCCESS;
}

/**
 * dsmil_simulated_smbios_init - Initialize simulated SMBIOS backend
 * @ctx: SMBIOS backend context
 *
 * Returns: 0 on success
 */
static inline int dsmil_simulated_smbios_init(struct dsmil_smbios_context *ctx)
{
	ctx->backend = DSMIL_BACKEND_SIMULATED;
	ctx->info.backend_type = DSMIL_BACKEND_SIMULATED;
	ctx->info.backend_name = "simulated (database-aware)";
	ctx->info.supports_token_read = true;
	ctx->info.supports_token_write = true;
	ctx->info.supports_token_discovery = true;
	ctx->info.supports_wmi = false;
	ctx->info.supports_smm = false;
	ctx->info.max_buffer_size = sizeof(struct calling_interface_buffer);

	pr_info("DSMIL: Initialized simulated SMBIOS backend\n");
	return 0;
}

/*
 * Unified SMBIOS Interface
 *
 * Provides a unified interface that automatically selects between:
 * 1. Real dell-smbios (when available and on Dell hardware)
 * 2. Simulated backend (fallback for all other cases)
 */

/**
 * dsmil_smbios_backend_init - Initialize SMBIOS backend
 * @ctx: SMBIOS backend context
 * @error_stats: Error statistics structure (optional)
 *
 * Automatically selects best available backend:
 * 1. Try real dell-smbios first (if CONFIG_DELL_SMBIOS=y)
 * 2. Fall back to simulated backend
 *
 * Returns: 0 on success, negative error code on failure
 */
static inline int dsmil_smbios_backend_init(struct dsmil_smbios_context *ctx,
					    struct dsmil_error_stats *error_stats)
{
	int ret;

	memset(ctx, 0, sizeof(*ctx));
	spin_lock_init(&ctx->lock);
	ctx->error_stats = error_stats;
	ctx->backend = DSMIL_BACKEND_NONE;

	/* Try real dell-smbios first */
	ret = dsmil_real_smbios_init(ctx);
	if (ret == 0) {
		pr_info("DSMIL: Using real dell-smbios backend\n");
		return 0;
	}

	/* Fall back to simulated backend */
	pr_info("DSMIL: Real dell-smbios unavailable, using simulated backend\n");
	return dsmil_simulated_smbios_init(ctx);
}

/**
 * dsmil_smbios_backend_call - Execute SMBIOS call via backend
 * @ctx: SMBIOS backend context
 * @buffer: SMBIOS call buffer
 *
 * Dispatches to appropriate backend (real or simulated)
 *
 * Returns: SMBIOS return code (SMBIOS_RET_SUCCESS, etc.)
 */
static inline int dsmil_smbios_backend_call(struct dsmil_smbios_context *ctx,
					    struct calling_interface_buffer *buffer)
{
	switch (ctx->backend) {
	case DSMIL_BACKEND_DELL_SMBIOS:
		return dsmil_real_smbios_call(ctx, buffer);

	case DSMIL_BACKEND_SIMULATED:
		return dsmil_simulated_smbios_call(ctx, buffer);

	default:
		pr_err("DSMIL: No SMBIOS backend initialized\n");
		return SMBIOS_RET_UNSUPPORTED_FUNC;
	}
}

/**
 * dsmil_smbios_backend_info - Get backend information
 * @ctx: SMBIOS backend context
 *
 * Returns: Pointer to backend info structure
 */
static inline const struct dsmil_smbios_backend_info *
dsmil_smbios_backend_info(struct dsmil_smbios_context *ctx)
{
	return &ctx->info;
}

#endif /* _DSMIL_REAL_SMBIOS_H */
