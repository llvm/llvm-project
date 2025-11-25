/*
 * DSMIL Error Handling & Logging Framework
 * ==========================================
 *
 * Comprehensive error handling, logging, and audit framework for DSMIL driver.
 *
 * Copyright (C) 2025 DSMIL Development Project
 * License: GPL v2
 */

#ifndef _DSMIL_ERROR_HANDLING_H
#define _DSMIL_ERROR_HANDLING_H

#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/ratelimit.h>

/*
 * Error Categories
 */
enum dsmil_error_category {
	DSMIL_ERR_NONE = 0,
	DSMIL_ERR_TOKEN,           /* Token operation errors */
	DSMIL_ERR_DEVICE,          /* Device operation errors */
	DSMIL_ERR_BIOS,            /* BIOS operation errors */
	DSMIL_ERR_AUTH,            /* Authentication errors */
	DSMIL_ERR_SECURITY,        /* Security violations */
	DSMIL_ERR_SMBIOS,          /* SMBIOS call errors */
	DSMIL_ERR_VALIDATION,      /* Validation errors */
	DSMIL_ERR_THERMAL,         /* Thermal errors */
	DSMIL_ERR_HARDWARE,        /* Hardware errors */
	DSMIL_ERR_SYSTEM,          /* System-level errors */
};

/*
 * Error Codes
 */
#define DSMIL_ERR_TOKEN_INVALID      1001
#define DSMIL_ERR_TOKEN_READONLY     1002
#define DSMIL_ERR_TOKEN_PROTECTED    1003
#define DSMIL_ERR_TOKEN_NOTFOUND     1004
#define DSMIL_ERR_TOKEN_VALIDATION   1005

#define DSMIL_ERR_DEVICE_INVALID     2001
#define DSMIL_ERR_DEVICE_OFFLINE     2002
#define DSMIL_ERR_DEVICE_ERROR       2003
#define DSMIL_ERR_DEVICE_LOCKED      2004
#define DSMIL_ERR_DEVICE_QUARANTINE  2005

#define DSMIL_ERR_BIOS_INVALID       3001
#define DSMIL_ERR_BIOS_CRITICAL      3002
#define DSMIL_ERR_BIOS_FAILOVER      3003
#define DSMIL_ERR_BIOS_SYNC          3004

#define DSMIL_ERR_AUTH_REQUIRED      4001
#define DSMIL_ERR_AUTH_FAILED        4002
#define DSMIL_ERR_AUTH_EXPIRED       4003
#define DSMIL_ERR_AUTH_INSUFFICIENT  4004

#define DSMIL_ERR_SECURITY_VIOLATION 5001
#define DSMIL_ERR_SECURITY_DENIED    5002
#define DSMIL_ERR_SECURITY_AUDIT     5003
#define DSMIL_ERR_SECURITY_TPM_FAILED 5004

#define DSMIL_ERR_SMBIOS_CALL        6001
#define DSMIL_ERR_SMBIOS_TIMEOUT     6002
#define DSMIL_ERR_SMBIOS_UNSUPPORTED 6003

#define DSMIL_ERR_THERMAL_WARNING    7001
#define DSMIL_ERR_THERMAL_CRITICAL   7002

/*
 * Error Information Structure
 */
struct dsmil_error_info {
	u32 error_code;
	enum dsmil_error_category category;
	const char *message;
	int linux_errno;  /* Corresponding Linux errno */
};

/*
 * Error Statistics
 */
struct dsmil_error_stats {
	atomic_t token_errors;
	atomic_t device_errors;
	atomic_t bios_errors;
	atomic_t auth_errors;
	atomic_t security_errors;
	atomic_t smbios_errors;
	atomic_t validation_errors;
	atomic_t thermal_errors;
	atomic_t total_errors;
	unsigned long last_error_time;
	u32 last_error_code;
};

/*
 * Audit Log Entry
 */
struct dsmil_audit_entry {
	ktime_t timestamp;
	u32 event_type;
	u32 user_id;
	u16 token_id;
	u32 old_value;
	u32 new_value;
	int result;
	char message[128];
};

/*
 * Audit Event Types
 */
#define DSMIL_AUDIT_TOKEN_READ       1
#define DSMIL_AUDIT_TOKEN_WRITE      2
#define DSMIL_AUDIT_PROTECTED_ACCESS 3
#define DSMIL_AUDIT_AUTH_SUCCESS     4
#define DSMIL_AUDIT_AUTH_FAILURE     5
#define DSMIL_AUDIT_BIOS_FAILOVER    6
#define DSMIL_AUDIT_BIOS_SYNC        7
#define DSMIL_AUDIT_SECURITY_EVENT   8
#define DSMIL_AUDIT_DEVICE_ACTIVATE  9
#define DSMIL_AUDIT_THERMAL_EVENT    10

/*
 * Error Database
 */
static const struct dsmil_error_info dsmil_error_db[] = {
	/* Token Errors */
	{
		.error_code = DSMIL_ERR_TOKEN_INVALID,
		.category = DSMIL_ERR_TOKEN,
		.message = "Invalid token ID",
		.linux_errno = -EINVAL,
	},
	{
		.error_code = DSMIL_ERR_TOKEN_READONLY,
		.category = DSMIL_ERR_TOKEN,
		.message = "Token is read-only",
		.linux_errno = -EPERM,
	},
	{
		.error_code = DSMIL_ERR_TOKEN_PROTECTED,
		.category = DSMIL_ERR_TOKEN,
		.message = "Token is protected (requires authentication)",
		.linux_errno = -EACCES,
	},
	{
		.error_code = DSMIL_ERR_TOKEN_NOTFOUND,
		.category = DSMIL_ERR_TOKEN,
		.message = "Token not found in database",
		.linux_errno = -ENOENT,
	},
	{
		.error_code = DSMIL_ERR_TOKEN_VALIDATION,
		.category = DSMIL_ERR_VALIDATION,
		.message = "Token value validation failed",
		.linux_errno = -EINVAL,
	},

	/* Device Errors */
	{
		.error_code = DSMIL_ERR_DEVICE_INVALID,
		.category = DSMIL_ERR_DEVICE,
		.message = "Invalid device ID",
		.linux_errno = -EINVAL,
	},
	{
		.error_code = DSMIL_ERR_DEVICE_OFFLINE,
		.category = DSMIL_ERR_DEVICE,
		.message = "Device is offline",
		.linux_errno = -ENXIO,
	},
	{
		.error_code = DSMIL_ERR_DEVICE_ERROR,
		.category = DSMIL_ERR_DEVICE,
		.message = "Device reported error",
		.linux_errno = -EIO,
	},
	{
		.error_code = DSMIL_ERR_DEVICE_QUARANTINE,
		.category = DSMIL_ERR_SECURITY,
		.message = "Device is quarantined",
		.linux_errno = -EPERM,
	},

	/* BIOS Errors */
	{
		.error_code = DSMIL_ERR_BIOS_INVALID,
		.category = DSMIL_ERR_BIOS,
		.message = "Invalid BIOS ID",
		.linux_errno = -EINVAL,
	},
	{
		.error_code = DSMIL_ERR_BIOS_CRITICAL,
		.category = DSMIL_ERR_BIOS,
		.message = "BIOS health critical",
		.linux_errno = -EIO,
	},
	{
		.error_code = DSMIL_ERR_BIOS_FAILOVER,
		.category = DSMIL_ERR_BIOS,
		.message = "BIOS failover failed",
		.linux_errno = -EIO,
	},

	/* Authentication Errors */
	{
		.error_code = DSMIL_ERR_AUTH_REQUIRED,
		.category = DSMIL_ERR_AUTH,
		.message = "Authentication required",
		.linux_errno = -EACCES,
	},
	{
		.error_code = DSMIL_ERR_AUTH_FAILED,
		.category = DSMIL_ERR_AUTH,
		.message = "Authentication failed",
		.linux_errno = -EPERM,
	},
	{
		.error_code = DSMIL_ERR_AUTH_EXPIRED,
		.category = DSMIL_ERR_AUTH,
		.message = "Authentication session expired",
		.linux_errno = -EACCES,
	},

	/* Security Errors */
	{
		.error_code = DSMIL_ERR_SECURITY_VIOLATION,
		.category = DSMIL_ERR_SECURITY,
		.message = "Security policy violation",
		.linux_errno = -EPERM,
	},

	/* SMBIOS Errors */
	{
		.error_code = DSMIL_ERR_SMBIOS_CALL,
		.category = DSMIL_ERR_SMBIOS,
		.message = "SMBIOS call failed",
		.linux_errno = -EIO,
	},

	/* Thermal Errors */
	{
		.error_code = DSMIL_ERR_THERMAL_WARNING,
		.category = DSMIL_ERR_THERMAL,
		.message = "Thermal warning threshold exceeded",
		.linux_errno = -ERANGE,
	},
	{
		.error_code = DSMIL_ERR_THERMAL_CRITICAL,
		.category = DSMIL_ERR_THERMAL,
		.message = "Thermal critical threshold exceeded",
		.linux_errno = -ERANGE,
	},

	/* Sentinel */
	{ .error_code = 0 }
};

/*
 * Error Lookup Function
 */
static inline const struct dsmil_error_info *dsmil_error_lookup(u32 error_code)
{
	int i;

	for (i = 0; dsmil_error_db[i].error_code != 0; i++) {
		if (dsmil_error_db[i].error_code == error_code)
			return &dsmil_error_db[i];
	}

	return NULL;
}

/*
 * Enhanced Logging Macros with Rate Limiting
 */
#define dsmil_err_ratelimited(fmt, ...) \
	pr_err_ratelimited("DSMIL: " fmt, ##__VA_ARGS__)

#define dsmil_warn_ratelimited(fmt, ...) \
	pr_warn_ratelimited("DSMIL: " fmt, ##__VA_ARGS__)

#define dsmil_info_ratelimited(fmt, ...) \
	pr_info_ratelimited("DSMIL: " fmt, ##__VA_ARGS__)

/*
 * Error Logging Macros
 */
#define dsmil_log_error(stats, category, code, fmt, ...) \
	do { \
		const struct dsmil_error_info *__err = dsmil_error_lookup(code); \
		atomic_inc(&(stats)->total_errors); \
		(stats)->last_error_code = code; \
		(stats)->last_error_time = jiffies; \
		if (__err) { \
			pr_err("DSMIL: [%s] %s: " fmt "\n", \
			       #category, __err->message, ##__VA_ARGS__); \
		} else { \
			pr_err("DSMIL: [%s] Error %u: " fmt "\n", \
			       #category, code, ##__VA_ARGS__); \
		} \
	} while (0)

#define dsmil_log_token_error(stats, code, token_id, fmt, ...) \
	do { \
		atomic_inc(&(stats)->token_errors); \
		dsmil_log_error(stats, TOKEN, code, \
				"token=0x%04x " fmt, token_id, ##__VA_ARGS__); \
	} while (0)

#define dsmil_log_device_error(stats, code, device_id, fmt, ...) \
	do { \
		atomic_inc(&(stats)->device_errors); \
		dsmil_log_error(stats, DEVICE, code, \
				"device=%u " fmt, device_id, ##__VA_ARGS__); \
	} while (0)

#define dsmil_log_bios_error(stats, code, bios_id, fmt, ...) \
	do { \
		atomic_inc(&(stats)->bios_errors); \
		dsmil_log_error(stats, BIOS, code, \
				"bios=%c " fmt, 'A' + bios_id, ##__VA_ARGS__); \
	} while (0)

#define dsmil_log_auth_error(stats, code, fmt, ...) \
	do { \
		atomic_inc(&(stats)->auth_errors); \
		dsmil_log_error(stats, AUTH, code, fmt, ##__VA_ARGS__); \
	} while (0)

#define dsmil_log_security_error(stats, code, fmt, ...) \
	do { \
		atomic_inc(&(stats)->security_errors); \
		dsmil_log_error(stats, SECURITY, code, fmt, ##__VA_ARGS__); \
	} while (0)

/*
 * Audit Logging Function
 */
static inline void dsmil_audit_log(struct dsmil_audit_entry *entry,
				   u32 event_type,
				   u16 token_id,
				   u32 old_value,
				   u32 new_value,
				   int result,
				   const char *message)
{
	entry->timestamp = ktime_get();
	entry->event_type = event_type;
	entry->user_id = from_kuid(&init_user_ns, current_uid());
	entry->token_id = token_id;
	entry->old_value = old_value;
	entry->new_value = new_value;
	entry->result = result;
	strscpy(entry->message, message, sizeof(entry->message));

	/* Log to kernel log */
	pr_info("DSMIL_AUDIT: type=%u uid=%u token=0x%04x old=0x%08x new=0x%08x result=%d msg=\"%s\"\n",
		event_type, entry->user_id, token_id, old_value, new_value,
		result, message);
}

/*
 * Error Recovery Hints
 */
static inline const char *dsmil_error_recovery_hint(u32 error_code)
{
	switch (error_code) {
	case DSMIL_ERR_TOKEN_PROTECTED:
		return "Authenticate using DSMIL_IOC_AUTHENTICATE before retrying";
	case DSMIL_ERR_AUTH_EXPIRED:
		return "Re-authenticate and retry operation";
	case DSMIL_ERR_DEVICE_OFFLINE:
		return "Wait for device to come online or check hardware connection";
	case DSMIL_ERR_BIOS_CRITICAL:
		return "Automatic failover should occur; monitor BIOS health";
	case DSMIL_ERR_THERMAL_CRITICAL:
		return "System overheating; check cooling system and reduce load";
	default:
		return "Check error details and retry if transient";
	}
}

/*
 * Error Statistics Helper Functions
 */
static inline void dsmil_error_stats_init(struct dsmil_error_stats *stats)
{
	atomic_set(&stats->token_errors, 0);
	atomic_set(&stats->device_errors, 0);
	atomic_set(&stats->bios_errors, 0);
	atomic_set(&stats->auth_errors, 0);
	atomic_set(&stats->security_errors, 0);
	atomic_set(&stats->smbios_errors, 0);
	atomic_set(&stats->validation_errors, 0);
	atomic_set(&stats->thermal_errors, 0);
	atomic_set(&stats->total_errors, 0);
	stats->last_error_time = 0;
	stats->last_error_code = 0;
}

static inline u32 dsmil_error_stats_total(const struct dsmil_error_stats *stats)
{
	return atomic_read(&stats->total_errors);
}

static inline bool dsmil_error_stats_recent(const struct dsmil_error_stats *stats,
					    unsigned long max_age_ms)
{
	if (stats->last_error_time == 0)
		return false;

	return time_before(jiffies, stats->last_error_time + msecs_to_jiffies(max_age_ms));
}

#endif /* _DSMIL_ERROR_HANDLING_H */
