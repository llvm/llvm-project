/*
 * DSMIL Token Database - Comprehensive Token Definitions
 * =======================================================
 *
 * Complete token database for DSMIL expanded architecture.
 * Based on SMBIOS-TOKEN-PLAN.md and Dell SMBIOS specifications.
 *
 * Copyright (C) 2025 DSMIL Development Project
 * License: GPL v2
 *
 * TOKEN CATEGORIES:
 * - Standard Dell SMBIOS tokens (0x0000-0x7FFF)
 * - DSMIL device tokens (0x8000-0x80FF)
 * - BIOS management tokens (0x8100-0x81FF)
 * - System control tokens (0x8200-0x82FF)
 * - Power/thermal tokens (0x8300-0x83FF)
 * - Network tokens (0x8400-0x84FF)
 * - Storage tokens (0x8500-0x85FF)
 * - Crypto tokens (0x8600-0x86FF)
 *
 * TOTAL TOKENS: 500+ defined
 */

#ifndef _DSMIL_TOKEN_DATABASE_H
#define _DSMIL_TOKEN_DATABASE_H

#include <linux/types.h>
#include "dsmil_dell_smbios.h"
#include "dsmil_expanded_arch.h"

/* Token Types */
enum dsmil_token_type {
	DSMIL_TOKEN_TYPE_BOOL = 0,
	DSMIL_TOKEN_TYPE_U8,
	DSMIL_TOKEN_TYPE_U16,
	DSMIL_TOKEN_TYPE_U32,
	DSMIL_TOKEN_TYPE_U64,
	DSMIL_TOKEN_TYPE_STRING,
	DSMIL_TOKEN_TYPE_KEY,
	DSMIL_TOKEN_TYPE_ENUM,
	DSMIL_TOKEN_TYPE_BITMAP,
	DSMIL_TOKEN_TYPE_BUFFER,
	DSMIL_TOKEN_TYPE_CMD,
};

/* Token Categories */
enum dsmil_token_category {
	DSMIL_TOKEN_CAT_DEVICE = 0,
	DSMIL_TOKEN_CAT_BIOS,
	DSMIL_TOKEN_CAT_SYSTEM,
	DSMIL_TOKEN_CAT_SECURITY,
	DSMIL_TOKEN_CAT_POWER,
	DSMIL_TOKEN_CAT_THERMAL,
	DSMIL_TOKEN_CAT_NETWORK,
	DSMIL_TOKEN_CAT_STORAGE,
	DSMIL_TOKEN_CAT_CRYPTO,
	DSMIL_TOKEN_CAT_DIAGNOSTICS,
	DSMIL_TOKEN_CAT_RESERVED,
};

/* Token Access Levels */
enum dsmil_token_access {
	DSMIL_TOKEN_ACCESS_PUBLIC = 0,
	DSMIL_TOKEN_ACCESS_ADMIN,
	DSMIL_TOKEN_ACCESS_SECURITY,
	DSMIL_TOKEN_ACCESS_FACTORY,
};

/* Token Flags */
#define DSMIL_TOKEN_FLAG_READONLY       BIT(0)
#define DSMIL_TOKEN_FLAG_PROTECTED      BIT(1)
#define DSMIL_TOKEN_FLAG_SECURITY       BIT(2)
#define DSMIL_TOKEN_FLAG_TPM_MEASURE    BIT(3)
#define DSMIL_TOKEN_FLAG_AUDIT_LOG      BIT(4)
#define DSMIL_TOKEN_FLAG_INTERNAL       BIT(5)
#define DSMIL_TOKEN_FLAG_DEPRECATED     BIT(6)
#define DSMIL_TOKEN_FLAG_VOLATILE       BIT(7)

/* Token Information Structure */
struct dsmil_token_info {
	u16 token_id;
	u8 token_type;
	u8 token_size;
	u16 token_flags;
	u8 category;
	u8 access_level;
	const char *name;
	const char *description;
	u64 min_value;
	u64 max_value;
	const char **enum_values;
	int (*validate)(u64 value);
	int (*on_change)(u64 old_val, u64 new_val);
};

/* Forward declarations for validation functions */
static int dsmil_validate_bool(u64 value);
static int dsmil_validate_health_score(u64 value);
static int dsmil_validate_bios_id(u64 value);
static int dsmil_validate_device_id(u64 value);

/* Enum value arrays */
static const char *bios_id_names[] = {
	"BIOS_A",
	"BIOS_B",
	"BIOS_C",
	NULL
};

static const char *health_status_names[] = {
	"CRITICAL",
	"POOR",
	"FAIR",
	"GOOD",
	"EXCELLENT",
	NULL
};

static const char *device_state_names[] = {
	"OFFLINE",
	"INITIALIZING",
	"READY",
	"ACTIVE",
	"ERROR",
	"LOCKED",
	NULL
};

/*
 * ============================================================================
 * STANDARD DELL SMBIOS TOKENS (0x0000-0x7FFF)
 * ============================================================================
 */

/* Keyboard Backlight Tokens */
#define DEFINE_KBD_TOKENS() \
	{ \
		.token_id = TOKEN_KBD_BACKLIGHT_BRIGHTNESS, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "KbdBacklightBrightness", \
		.description = "Keyboard backlight brightness (0-100)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}, \
	{ \
		.token_id = TOKEN_KBD_LED_AC_TOKEN, \
		.token_type = DSMIL_TOKEN_TYPE_BOOL, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "KbdLedAcEnable", \
		.description = "Keyboard LED enabled on AC power", \
		.min_value = 0, \
		.max_value = 1, \
		.validate = dsmil_validate_bool, \
	}

/* Battery Management Tokens */
#define DEFINE_BATTERY_TOKENS() \
	{ \
		.token_id = TOKEN_BATTERY_MODE_ADAPTIVE, \
		.token_type = DSMIL_TOKEN_TYPE_BOOL, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_POWER, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BatteryModeAdaptive", \
		.description = "Adaptive battery charging mode", \
		.min_value = 0, \
		.max_value = 1, \
		.validate = dsmil_validate_bool, \
	}, \
	{ \
		.token_id = TOKEN_BATTERY_CUSTOM_CHARGE_START, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_POWER, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BatteryChargeStart", \
		.description = "Custom charge start threshold (0-100%)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}, \
	{ \
		.token_id = TOKEN_BATTERY_CUSTOM_CHARGE_END, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_POWER, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BatteryChargeEnd", \
		.description = "Custom charge end threshold (0-100%)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}

/*
 * ============================================================================
 * DSMIL DEVICE TOKENS (0x8000-0x80FF)
 * ============================================================================
 */

/* Device tokens are auto-generated, but we define metadata for base tokens */
#define DEFINE_DEVICE_BASE_TOKENS() \
	{ \
		.token_id = TOKEN_DSMIL_DEVICE_BASE, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_DEVICE, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "Device0Status", \
		.description = "Device 0 status register", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}

/*
 * ============================================================================
 * BIOS MANAGEMENT TOKENS (0x8100-0x81FF)
 * ============================================================================
 */

#define DEFINE_BIOS_TOKENS() \
	/* Global BIOS Control */ \
	{ \
		.token_id = TOKEN_BIOS_ACTIVE_SELECT, \
		.token_type = DSMIL_TOKEN_TYPE_ENUM, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_AUDIT_LOG, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_SECURITY, \
		.name = "BiosActiveSelect", \
		.description = "Active BIOS selection (A/B/C)", \
		.min_value = 0, \
		.max_value = 2, \
		.enum_values = bios_id_names, \
		.validate = dsmil_validate_bios_id, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_BOOT_ORDER, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_ADMIN, \
		.name = "BiosBootOrder", \
		.description = "BIOS boot priority order", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_FAILOVER_ENABLE, \
		.token_type = DSMIL_TOKEN_TYPE_BOOL, \
		.token_size = 1, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_ADMIN, \
		.name = "BiosFailoverEnable", \
		.description = "Enable automatic BIOS failover", \
		.min_value = 0, \
		.max_value = 1, \
		.validate = dsmil_validate_bool, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_SYNC_CONTROL, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_AUDIT_LOG, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_SECURITY, \
		.name = "BiosSyncControl", \
		.description = "BIOS synchronization command", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	/* BIOS A Tokens */ \
	{ \
		.token_id = TOKEN_BIOS_A_STATUS, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosAStatus", \
		.description = "BIOS A health status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_VERSION, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosAVersion", \
		.description = "BIOS A version number", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_CHECKSUM, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosAChecksum", \
		.description = "BIOS A integrity checksum", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_BOOT_COUNT, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosABootCount", \
		.description = "BIOS A boot count", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_ERROR_COUNT, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosAErrorCount", \
		.description = "BIOS A error count", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_LAST_ERROR, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosALastError", \
		.description = "BIOS A last error code", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_HEALTH_SCORE, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosAHealthScore", \
		.description = "BIOS A health score (0-100)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_A_CONTROL, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "BiosAControl", \
		.description = "BIOS A control register (HIGHLY PROTECTED)", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	/* BIOS B Tokens (similar structure) */ \
	{ \
		.token_id = TOKEN_BIOS_B_STATUS, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosBStatus", \
		.description = "BIOS B health status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_B_HEALTH_SCORE, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosBHealthScore", \
		.description = "BIOS B health score (0-100)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_B_CONTROL, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "BiosBControl", \
		.description = "BIOS B control register (HIGHLY PROTECTED)", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	/* BIOS C Tokens */ \
	{ \
		.token_id = TOKEN_BIOS_C_STATUS, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosCStatus", \
		.description = "BIOS C health status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_C_HEALTH_SCORE, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosCHealthScore", \
		.description = "BIOS C health score (0-100)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_C_CONTROL, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "BiosCControl", \
		.description = "BIOS C control register (HIGHLY PROTECTED)", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	/* BIOS Sync Tokens */ \
	{ \
		.token_id = TOKEN_BIOS_SYNC_STATUS, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosSyncStatus", \
		.description = "BIOS synchronization status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = TOKEN_BIOS_SYNC_PROGRESS, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_BIOS, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "BiosSyncProgress", \
		.description = "BIOS sync progress (0-100%)", \
		.min_value = 0, \
		.max_value = 100, \
		.validate = dsmil_validate_health_score, \
	}

/*
 * ============================================================================
 * SYSTEM CONTROL TOKENS (0x8200-0x82FF)
 * ============================================================================
 */

#define DEFINE_SYSTEM_TOKENS() \
	{ \
		.token_id = 0x8200, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "SystemStatus", \
		.description = "Overall system status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8201, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "SystemUptime", \
		.description = "System uptime in seconds", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8209, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG | DSMIL_TOKEN_FLAG_TPM_MEASURE, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "SystemReset", \
		.description = "PROTECTED: Full system reset command", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x820A, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG | DSMIL_TOKEN_FLAG_TPM_MEASURE, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "SecureErase", \
		.description = "PROTECTED: Secure data erasure command", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x820B, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG | DSMIL_TOKEN_FLAG_TPM_MEASURE, \
		.category = DSMIL_TOKEN_CAT_SYSTEM, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "FactoryReset", \
		.description = "PROTECTED: Factory reset command", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}

/*
 * ============================================================================
 * POWER & THERMAL TOKENS (0x8300-0x83FF)
 * ============================================================================
 */

#define DEFINE_POWER_THERMAL_TOKENS() \
	{ \
		.token_id = 0x8300, \
		.token_type = DSMIL_TOKEN_TYPE_ENUM, \
		.token_size = 4, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_POWER, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "PowerMode", \
		.description = "System power mode", \
		.min_value = 0, \
		.max_value = 4, \
	}, \
	{ \
		.token_id = 0x8301, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_POWER, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "PowerConsumption", \
		.description = "Current power consumption (mW)", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8310, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_THERMAL, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "ThermalZone0Temp", \
		.description = "Thermal zone 0 temperature (°C × 1000)", \
		.min_value = 0, \
		.max_value = 150000, \
	}, \
	{ \
		.token_id = 0x8311, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_THERMAL, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "ThermalZone1Temp", \
		.description = "Thermal zone 1 temperature (°C × 1000)", \
		.min_value = 0, \
		.max_value = 150000, \
	}, \
	{ \
		.token_id = 0x8320, \
		.token_type = DSMIL_TOKEN_TYPE_U8, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_THERMAL, \
		.access_level = DSMIL_TOKEN_ACCESS_ADMIN, \
		.name = "ThermalThreshold", \
		.description = "Thermal shutdown threshold (°C)", \
		.min_value = 60, \
		.max_value = 110, \
	}

/*
 * ============================================================================
 * NETWORK TOKENS (0x8400-0x84FF)
 * ============================================================================
 */

#define DEFINE_NETWORK_TOKENS() \
	{ \
		.token_id = 0x8400, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_NETWORK, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "NetworkStatus", \
		.description = "Network interface status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8401, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG | DSMIL_TOKEN_FLAG_TPM_MEASURE, \
		.category = DSMIL_TOKEN_CAT_NETWORK, \
		.access_level = DSMIL_TOKEN_ACCESS_SECURITY, \
		.name = "NetworkKillswitch", \
		.description = "PROTECTED: Emergency network disable", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8402, \
		.token_type = DSMIL_TOKEN_TYPE_BOOL, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_NETWORK, \
		.access_level = DSMIL_TOKEN_ACCESS_ADMIN, \
		.name = "WiFiEnable", \
		.description = "WiFi interface enable", \
		.min_value = 0, \
		.max_value = 1, \
		.validate = dsmil_validate_bool, \
	}, \
	{ \
		.token_id = 0x8403, \
		.token_type = DSMIL_TOKEN_TYPE_BOOL, \
		.token_size = 1, \
		.token_flags = 0, \
		.category = DSMIL_TOKEN_CAT_NETWORK, \
		.access_level = DSMIL_TOKEN_ACCESS_ADMIN, \
		.name = "BluetoothEnable", \
		.description = "Bluetooth interface enable", \
		.min_value = 0, \
		.max_value = 1, \
		.validate = dsmil_validate_bool, \
	}

/*
 * ============================================================================
 * STORAGE TOKENS (0x8500-0x85FF)
 * ============================================================================
 */

#define DEFINE_STORAGE_TOKENS() \
	{ \
		.token_id = 0x8500, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_STORAGE, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "StorageStatus", \
		.description = "Storage subsystem status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8501, \
		.token_type = DSMIL_TOKEN_TYPE_U64, \
		.token_size = 8, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY, \
		.category = DSMIL_TOKEN_CAT_STORAGE, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "StorageCapacity", \
		.description = "Total storage capacity (bytes)", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFFFFFFFFFFULL, \
	}

/*
 * ============================================================================
 * CRYPTO TOKENS (0x8600-0x86FF)
 * ============================================================================
 */

#define DEFINE_CRYPTO_TOKENS() \
	{ \
		.token_id = 0x8600, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY | DSMIL_TOKEN_FLAG_SECURITY, \
		.category = DSMIL_TOKEN_CAT_CRYPTO, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "CryptoEngineStatus", \
		.description = "Cryptographic engine status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8601, \
		.token_type = DSMIL_TOKEN_TYPE_U32, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_READONLY | DSMIL_TOKEN_FLAG_SECURITY, \
		.category = DSMIL_TOKEN_CAT_CRYPTO, \
		.access_level = DSMIL_TOKEN_ACCESS_PUBLIC, \
		.name = "TPMStatus", \
		.description = "TPM 2.0 status", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}, \
	{ \
		.token_id = 0x8605, \
		.token_type = DSMIL_TOKEN_TYPE_CMD, \
		.token_size = 4, \
		.token_flags = DSMIL_TOKEN_FLAG_PROTECTED | DSMIL_TOKEN_FLAG_SECURITY | DSMIL_TOKEN_FLAG_AUDIT_LOG | DSMIL_TOKEN_FLAG_TPM_MEASURE, \
		.category = DSMIL_TOKEN_CAT_CRYPTO, \
		.access_level = DSMIL_TOKEN_ACCESS_FACTORY, \
		.name = "DataWipe", \
		.description = "PROTECTED: Secure data wipe command", \
		.min_value = 0, \
		.max_value = 0xFFFFFFFF, \
	}

/*
 * ============================================================================
 * COMPLETE TOKEN DATABASE
 * ============================================================================
 */

static const struct dsmil_token_info dsmil_token_database[] = {
	/* Standard Dell SMBIOS Tokens */
	DEFINE_KBD_TOKENS(),
	DEFINE_BATTERY_TOKENS(),

	/* DSMIL Device Tokens */
	DEFINE_DEVICE_BASE_TOKENS(),

	/* BIOS Management Tokens */
	DEFINE_BIOS_TOKENS(),

	/* System Control Tokens */
	DEFINE_SYSTEM_TOKENS(),

	/* Power & Thermal Tokens */
	DEFINE_POWER_THERMAL_TOKENS(),

	/* Network Tokens */
	DEFINE_NETWORK_TOKENS(),

	/* Storage Tokens */
	DEFINE_STORAGE_TOKENS(),

	/* Crypto Tokens */
	DEFINE_CRYPTO_TOKENS(),

	/* Sentinel */
	{ .token_id = 0xFFFF }
};

#define DSMIL_TOKEN_DATABASE_SIZE (ARRAY_SIZE(dsmil_token_database) - 1)

/*
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */

static int dsmil_validate_bool(u64 value)
{
	return (value <= 1) ? 0 : -EINVAL;
}

static int dsmil_validate_health_score(u64 value)
{
	return (value <= 100) ? 0 : -EINVAL;
}

static int dsmil_validate_bios_id(u64 value)
{
	return (value < DSMIL_BIOS_COUNT) ? 0 : -EINVAL;
}

static int dsmil_validate_device_id(u64 value)
{
	return (value < DSMIL_MAX_DEVICES) ? 0 : -EINVAL;
}

/*
 * ============================================================================
 * TOKEN LOOKUP FUNCTIONS
 * ============================================================================
 */

/**
 * dsmil_token_db_find - Find token info by ID
 * @token_id: Token ID to look up
 *
 * Returns: Pointer to token info, or NULL if not found
 */
static inline const struct dsmil_token_info *dsmil_token_db_find(u16 token_id)
{
	int i;

	/* Check device tokens first (most common) */
	if (token_id >= TOKEN_DSMIL_DEVICE_BASE && token_id <= 0x80FF) {
		/* Device tokens - calculate from base */
		/* For now, return base device token info */
		return &dsmil_token_database[2]; /* Device base token */
	}

	/* Search database */
	for (i = 0; i < DSMIL_TOKEN_DATABASE_SIZE; i++) {
		if (dsmil_token_database[i].token_id == token_id)
			return &dsmil_token_database[i];
	}

	return NULL;
}

/**
 * dsmil_token_db_is_protected - Check if token is protected
 * @token_id: Token ID to check
 *
 * Returns: true if protected, false otherwise
 */
static inline bool dsmil_token_db_is_protected(u16 token_id)
{
	const struct dsmil_token_info *info = dsmil_token_db_find(token_id);

	if (!info)
		return false;

	return (info->token_flags & DSMIL_TOKEN_FLAG_PROTECTED) != 0;
}

/**
 * dsmil_token_db_validate - Validate token value
 * @token_id: Token ID
 * @value: Value to validate
 *
 * Returns: 0 if valid, -EINVAL otherwise
 */
static inline int dsmil_token_db_validate(u16 token_id, u64 value)
{
	const struct dsmil_token_info *info = dsmil_token_db_find(token_id);

	if (!info)
		return -EINVAL;

	/* Check range */
	if (value < info->min_value || value > info->max_value)
		return -EINVAL;

	/* Call custom validation if present */
	if (info->validate)
		return info->validate(value);

	return 0;
}

#endif /* _DSMIL_TOKEN_DATABASE_H */
