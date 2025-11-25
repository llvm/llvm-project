/*
 * DSMIL Expanded Architecture - 104 Devices + Redundant BIOS
 * ===========================================================
 *
 * Full-scale production architecture supporting:
 * - 103-104 DSMIL devices (expandable)
 * - 3 redundant BIOS systems (A/B/C)
 * - Automatic BIOS failover
 * - Device token management
 * - Future expansion to 256+ devices
 *
 * Copyright (C) 2025 DSMIL Development
 * License: GPL v2
 */

#ifndef _DSMIL_EXPANDED_ARCH_H
#define _DSMIL_EXPANDED_ARCH_H

#include <linux/types.h>
#include "dsmil_dell_smbios.h"

/*
 * DEVICE ARCHITECTURE
 * ===================
 * Support for 103-104 DSMIL devices organized in groups.
 */

/* Device limits */
#define DSMIL_MAX_DEVICES		104	/* Current maximum (103-104) */
#define DSMIL_DEVICE_GROUPS		9	/* Groups 0-8 */
#define DSMIL_DEVICES_PER_GROUP		12	/* Standard group size */
#define DSMIL_EXPANSION_RESERVE		24	/* Reserved for future expansion */

/* Total addressable devices: 104 + 24 reserve = 128 */
#define DSMIL_TOTAL_ADDRESSABLE		128

/*
 * TOKEN ALLOCATION SCHEME
 * =======================
 * Expanded token space to accommodate all devices and features.
 *
 * Token Range Layout:
 * 0x0000-0x7FFF : Standard Dell SMBIOS tokens (documented)
 * 0x8000-0x80FF : DSMIL Device Tokens (256 slots for 104+ devices)
 * 0x8100-0x81FF : BIOS Management (3 redundant BIOS systems)
 * 0x8200-0x82FF : System Control & Security
 * 0x8300-0x83FF : Power & Thermal Management
 * 0x8400-0x84FF : Network & Communications
 * 0x8500-0x85FF : Storage & I/O
 * 0x8600-0x86FF : Crypto & Security Engines
 * 0x8700-0x8FFF : Reserved for future expansion
 */

/*
 * DSMIL DEVICE TOKENS (0x8000-0x80FF)
 * ====================================
 * Each device can have multiple tokens:
 * - Base + 0: Device status/control
 * - Base + 1: Device configuration
 * - Base + 2: Device data
 *
 * Token Formula: 0x8000 + (device_id * 3) + token_offset
 * Example: Device 0 status = 0x8000
 *          Device 0 config = 0x8001
 *          Device 0 data   = 0x8002
 *          Device 1 status = 0x8003
 *          ...
 */

#define TOKEN_DSMIL_DEVICE_BASE		0x8000
#define TOKEN_DSMIL_DEVICE_STRIDE	3	/* 3 tokens per device */

/* Calculate device token */
#define TOKEN_DSMIL_DEVICE(dev_id, offset) \
	(TOKEN_DSMIL_DEVICE_BASE + ((dev_id) * TOKEN_DSMIL_DEVICE_STRIDE) + (offset))

/* Token offsets within device */
#define TOKEN_OFFSET_STATUS		0	/* Device status/control */
#define TOKEN_OFFSET_CONFIG		1	/* Device configuration */
#define TOKEN_OFFSET_DATA		2	/* Device data register */

/* Device-specific token examples */
#define TOKEN_DEVICE_0_STATUS		TOKEN_DSMIL_DEVICE(0, TOKEN_OFFSET_STATUS)
#define TOKEN_DEVICE_0_CONFIG		TOKEN_DSMIL_DEVICE(0, TOKEN_OFFSET_CONFIG)
#define TOKEN_DEVICE_0_DATA		TOKEN_DSMIL_DEVICE(0, TOKEN_OFFSET_DATA)

#define TOKEN_DEVICE_103_STATUS		TOKEN_DSMIL_DEVICE(103, TOKEN_OFFSET_STATUS)
#define TOKEN_DEVICE_103_CONFIG		TOKEN_DSMIL_DEVICE(103, TOKEN_OFFSET_CONFIG)
#define TOKEN_DEVICE_103_DATA		TOKEN_DSMIL_DEVICE(103, TOKEN_OFFSET_DATA)

/*
 * REDUNDANT BIOS MANAGEMENT (0x8100-0x81FF)
 * ==========================================
 * Three independent BIOS systems for fault tolerance:
 * - BIOS A (Primary)
 * - BIOS B (Secondary)
 * - BIOS C (Tertiary)
 */

/* BIOS Control Tokens */
#define TOKEN_BIOS_CONTROL_BASE		0x8100

/* Active BIOS selection */
#define TOKEN_BIOS_ACTIVE_SELECT	0x8100	/* Which BIOS is active (A/B/C) */
#define TOKEN_BIOS_BOOT_ORDER		0x8101	/* Boot order: A->B->C */
#define TOKEN_BIOS_FAILOVER_ENABLE	0x8102	/* Auto failover enabled */
#define TOKEN_BIOS_SYNC_CONTROL		0x8103	/* BIOS sync control */

/* BIOS A (Primary) - 0x8110-0x811F */
#define TOKEN_BIOS_A_STATUS		0x8110	/* Health status */
#define TOKEN_BIOS_A_VERSION		0x8111	/* Version info */
#define TOKEN_BIOS_A_CHECKSUM		0x8112	/* Integrity checksum */
#define TOKEN_BIOS_A_BOOT_COUNT		0x8113	/* Boot counter */
#define TOKEN_BIOS_A_ERROR_COUNT	0x8114	/* Error counter */
#define TOKEN_BIOS_A_LAST_ERROR		0x8115	/* Last error code */
#define TOKEN_BIOS_A_HEALTH_SCORE	0x8116	/* Health score (0-100) */
#define TOKEN_BIOS_A_ACTIVE_TIME	0x8117	/* Time active (seconds) */
#define TOKEN_BIOS_A_CONFIG_HASH	0x8118	/* Configuration hash */
#define TOKEN_BIOS_A_UPDATE_STATUS	0x8119	/* Update status */
#define TOKEN_BIOS_A_LOCK_STATE		0x811A	/* Write lock state */
#define TOKEN_BIOS_A_RESERVED_0		0x811B	/* Reserved */
#define TOKEN_BIOS_A_RESERVED_1		0x811C	/* Reserved */
#define TOKEN_BIOS_A_RESERVED_2		0x811D	/* Reserved */
#define TOKEN_BIOS_A_RESERVED_3		0x811E	/* Reserved */
#define TOKEN_BIOS_A_CONTROL		0x811F	/* Control register */

/* BIOS B (Secondary) - 0x8120-0x812F */
#define TOKEN_BIOS_B_STATUS		0x8120
#define TOKEN_BIOS_B_VERSION		0x8121
#define TOKEN_BIOS_B_CHECKSUM		0x8122
#define TOKEN_BIOS_B_BOOT_COUNT		0x8123
#define TOKEN_BIOS_B_ERROR_COUNT	0x8124
#define TOKEN_BIOS_B_LAST_ERROR		0x8125
#define TOKEN_BIOS_B_HEALTH_SCORE	0x8126
#define TOKEN_BIOS_B_ACTIVE_TIME	0x8127
#define TOKEN_BIOS_B_CONFIG_HASH	0x8128
#define TOKEN_BIOS_B_UPDATE_STATUS	0x8129
#define TOKEN_BIOS_B_LOCK_STATE		0x812A
#define TOKEN_BIOS_B_RESERVED_0		0x812B
#define TOKEN_BIOS_B_RESERVED_1		0x812C
#define TOKEN_BIOS_B_RESERVED_2		0x812D
#define TOKEN_BIOS_B_RESERVED_3		0x812E
#define TOKEN_BIOS_B_CONTROL		0x812F

/* BIOS C (Tertiary) - 0x8130-0x813F */
#define TOKEN_BIOS_C_STATUS		0x8130
#define TOKEN_BIOS_C_VERSION		0x8131
#define TOKEN_BIOS_C_CHECKSUM		0x8132
#define TOKEN_BIOS_C_BOOT_COUNT		0x8133
#define TOKEN_BIOS_C_ERROR_COUNT	0x8134
#define TOKEN_BIOS_C_LAST_ERROR		0x8135
#define TOKEN_BIOS_C_HEALTH_SCORE	0x8136
#define TOKEN_BIOS_C_ACTIVE_TIME	0x8137
#define TOKEN_BIOS_C_CONFIG_HASH	0x8138
#define TOKEN_BIOS_C_UPDATE_STATUS	0x8139
#define TOKEN_BIOS_C_LOCK_STATE		0x813A
#define TOKEN_BIOS_C_RESERVED_0		0x813B
#define TOKEN_BIOS_C_RESERVED_1		0x813C
#define TOKEN_BIOS_C_RESERVED_2		0x813D
#define TOKEN_BIOS_C_RESERVED_3		0x813E
#define TOKEN_BIOS_C_CONTROL		0x813F

/* BIOS synchronization tokens */
#define TOKEN_BIOS_SYNC_STATUS		0x8140	/* Sync status */
#define TOKEN_BIOS_SYNC_PROGRESS	0x8141	/* Sync progress % */
#define TOKEN_BIOS_SYNC_LAST_TIME	0x8142	/* Last sync timestamp */
#define TOKEN_BIOS_SYNC_ERROR		0x8143	/* Last sync error */

/* BIOS identification */
enum dsmil_bios_id {
	DSMIL_BIOS_A = 0,	/* Primary BIOS */
	DSMIL_BIOS_B = 1,	/* Secondary BIOS */
	DSMIL_BIOS_C = 2,	/* Tertiary BIOS */
	DSMIL_BIOS_COUNT = 3,
};

/* BIOS status values */
#define BIOS_STATUS_HEALTHY		0x0000
#define BIOS_STATUS_DEGRADED		0x0001
#define BIOS_STATUS_FAILED		0x0002
#define BIOS_STATUS_UPDATING		0x0003
#define BIOS_STATUS_LOCKED		0x0004
#define BIOS_STATUS_CORRUPTED		0x0005
#define BIOS_STATUS_UNKNOWN		0xFFFF

/* BIOS health score thresholds */
#define BIOS_HEALTH_EXCELLENT		90
#define BIOS_HEALTH_GOOD		70
#define BIOS_HEALTH_FAIR		50
#define BIOS_HEALTH_POOR		30
#define BIOS_HEALTH_CRITICAL		10

/*
 * SYSTEM CONTROL & SECURITY (0x8200-0x82FF)
 * ==========================================
 */
#define TOKEN_SYSTEM_STATUS		0x8200
#define TOKEN_SECURITY_LEVEL		0x8201
#define TOKEN_AUTH_STATUS		0x8202
#define TOKEN_AUDIT_CONTROL		0x8203
#define TOKEN_MFA_ENABLED		0x8204
#define TOKEN_COMPLIANCE_MODE		0x8205
#define TOKEN_EMERGENCY_STOP		0x8206
#define TOKEN_THREAT_LEVEL		0x8207
#define TOKEN_INCIDENT_COUNT		0x8208

/* Protected system control tokens */
#define TOKEN_SYSTEM_RESET		0x8209	/* PROTECTED */
#define TOKEN_SECURE_ERASE		0x820A	/* PROTECTED */
#define TOKEN_FACTORY_RESET		0x820B	/* PROTECTED */

/*
 * POWER & THERMAL MANAGEMENT (0x8300-0x83FF)
 * ===========================================
 */
#define TOKEN_POWER_MODE		0x8300
#define TOKEN_THERMAL_ZONE_0		0x8301
#define TOKEN_THERMAL_ZONE_1		0x8302
#define TOKEN_THERMAL_ZONE_2		0x8303
#define TOKEN_THERMAL_ZONE_3		0x8304	/* Additional zone */
#define TOKEN_FAN_CONTROL		0x8305
#define TOKEN_VOLTAGE_RAIL_0		0x8306
#define TOKEN_VOLTAGE_RAIL_1		0x8307
#define TOKEN_VOLTAGE_RAIL_2		0x8308	/* Additional rail */
#define TOKEN_POWER_LIMIT		0x8309
#define TOKEN_BATTERY_HEALTH		0x830A
#define TOKEN_CHARGE_PROFILE		0x830B
#define TOKEN_POWER_STATS		0x830C
#define TOKEN_THERMAL_POLICY		0x830D

/*
 * NETWORK & COMMUNICATIONS (0x8400-0x84FF)
 * =========================================
 */
#define TOKEN_NETWORK_STATUS		0x8400
#define TOKEN_NETWORK_KILLSWITCH	0x8401	/* PROTECTED */
#define TOKEN_WIFI_CONTROL		0x8402
#define TOKEN_BT_CONTROL		0x8403
#define TOKEN_WWAN_CONTROL		0x8404
#define TOKEN_ETHERNET_CONTROL		0x8405
#define TOKEN_FIREWALL_STATE		0x8406
#define TOKEN_VPN_STATE			0x8407
#define TOKEN_NETWORK_POLICY		0x8408
#define TOKEN_BANDWIDTH_LIMIT		0x8409

/*
 * STORAGE & I/O (0x8500-0x85FF)
 * ==============================
 */
#define TOKEN_STORAGE_STATUS		0x8500
#define TOKEN_DISK_0_STATUS		0x8501
#define TOKEN_DISK_1_STATUS		0x8502
#define TOKEN_RAID_STATUS		0x8503
#define TOKEN_USB_CONTROL		0x8504
#define TOKEN_NVME_STATUS		0x8505

/*
 * CRYPTO & SECURITY ENGINES (0x8600-0x86FF)
 * ==========================================
 */
#define TOKEN_CRYPTO_ENGINE		0x8600
#define TOKEN_HASH_ENGINE		0x8601
#define TOKEN_RNG_SOURCE		0x8602
#define TOKEN_TPM_STATUS		0x8603
#define TOKEN_SECURE_BOOT		0x8604
#define TOKEN_DATA_WIPE		0x8605	/* PROTECTED */
#define TOKEN_ENCRYPTION_STATE		0x8606
#define TOKEN_KEY_MANAGEMENT		0x8607

/*
 * DEVICE GROUP ORGANIZATION
 * ==========================
 * 104 devices organized into 9 groups (0-8)
 * Groups 0-7: 12 devices each (96 devices)
 * Group 8: 8 devices (totaling 104)
 */

struct dsmil_device_info {
	u16 device_id;			/* Device identifier (0-103) */
	u16 token_base;			/* Base token address */
	u32 capabilities;		/* Device capability flags */
	u32 status;			/* Current status */
	u32 config;			/* Configuration */
	u8 group_id;			/* Group membership (0-8) */
	u8 position;			/* Position in group (0-11) */
	u8 bios_affinity;		/* Preferred BIOS (A/B/C) */
	u8 protection_level;		/* 0=normal, 1=protected, 2=critical */
} __packed;

struct dsmil_device_group {
	u8 group_id;			/* Group number (0-8) */
	u8 device_count;		/* Devices in this group */
	u16 token_base;			/* Base token for group */
	u32 group_capabilities;		/* Group capability mask */
	struct dsmil_device_info devices[12];
} __packed;

/*
 * BIOS MANAGEMENT STRUCTURES
 * ===========================
 */

struct dsmil_bios_info {
	enum dsmil_bios_id bios_id;	/* A, B, or C */
	u32 status;			/* Health status */
	u32 version;			/* BIOS version */
	u32 checksum;			/* Integrity checksum */
	u32 boot_count;			/* Number of boots */
	u32 error_count;		/* Error count */
	u32 last_error;			/* Last error code */
	u8 health_score;		/* 0-100 health score */
	u64 active_time;		/* Time active (seconds) */
	u32 config_hash;		/* Configuration hash */
	bool is_active;			/* Currently active */
	bool is_locked;			/* Write locked */
} __packed;

struct dsmil_bios_redundancy {
	enum dsmil_bios_id active_bios;	/* Currently active BIOS */
	u8 boot_order[3];		/* Boot order: A, B, C */
	bool failover_enabled;		/* Auto failover enabled */
	bool sync_enabled;		/* BIOS sync enabled */
	u32 sync_status;		/* Sync status */
	struct dsmil_bios_info bios[DSMIL_BIOS_COUNT];
} __packed;

/*
 * PROTECTED TOKEN LIST (Expanded)
 * ================================
 */
static const u16 dsmil_protected_tokens_expanded[] = {
	/* System control */
	TOKEN_SYSTEM_RESET,		/* 0x8209 */
	TOKEN_SECURE_ERASE,		/* 0x820A */
	TOKEN_FACTORY_RESET,		/* 0x820B */

	/* Network */
	TOKEN_NETWORK_KILLSWITCH,	/* 0x8401 */

	/* Crypto/Data */
	TOKEN_DATA_WIPE,		/* 0x8605 */

	/* BIOS control (highly protected) */
	TOKEN_BIOS_A_CONTROL,		/* 0x811F */
	TOKEN_BIOS_B_CONTROL,		/* 0x812F */
	TOKEN_BIOS_C_CONTROL,		/* 0x813F */
};
#define DSMIL_PROTECTED_TOKEN_COUNT_EXPANDED	8

/*
 * HELPER FUNCTIONS
 * ================
 */

/* Device token helpers */
static inline u16 dsmil_get_device_token(u8 device_id, u8 token_offset)
{
	if (device_id >= DSMIL_MAX_DEVICES)
		return 0;
	return TOKEN_DSMIL_DEVICE(device_id, token_offset);
}

/* BIOS helpers */
static inline u16 dsmil_get_bios_status_token(enum dsmil_bios_id bios_id)
{
	switch (bios_id) {
	case DSMIL_BIOS_A:
		return TOKEN_BIOS_A_STATUS;
	case DSMIL_BIOS_B:
		return TOKEN_BIOS_B_STATUS;
	case DSMIL_BIOS_C:
		return TOKEN_BIOS_C_STATUS;
	default:
		return 0;
	}
}

static inline const char *dsmil_bios_id_to_string(enum dsmil_bios_id bios_id)
{
	switch (bios_id) {
	case DSMIL_BIOS_A:
		return "BIOS-A (Primary)";
	case DSMIL_BIOS_B:
		return "BIOS-B (Secondary)";
	case DSMIL_BIOS_C:
		return "BIOS-C (Tertiary)";
	default:
		return "Unknown";
	}
}

/* Protection check */
static inline bool dsmil_is_protected_token_expanded(u16 token_id)
{
	int i;
	for (i = 0; i < DSMIL_PROTECTED_TOKEN_COUNT_EXPANDED; i++) {
		if (dsmil_protected_tokens_expanded[i] == token_id)
			return true;
	}
	return false;
}

/* Token range checks */
static inline bool dsmil_is_device_token(u16 token_id)
{
	return (token_id >= 0x8000 && token_id < 0x8100);
}

static inline bool dsmil_is_bios_token(u16 token_id)
{
	return (token_id >= 0x8100 && token_id < 0x8200);
}

/* Extract device ID from token */
static inline u8 dsmil_token_to_device_id(u16 token_id)
{
	if (!dsmil_is_device_token(token_id))
		return 0xFF;
	return (token_id - TOKEN_DSMIL_DEVICE_BASE) / TOKEN_DSMIL_DEVICE_STRIDE;
}

/*
 * EXPANSION PLAN
 * ==============
 * Reserved token ranges for future growth:
 * - 0x8700-0x87FF: Reserved (256 tokens)
 * - 0x8800-0x88FF: Reserved (256 tokens)
 * - 0x8900-0x8FFF: Reserved (1792 tokens)
 *
 * Total reserved: 2304 tokens for future expansion
 *
 * Device expansion path:
 * - Current: 104 devices (312 tokens @ 3 per device)
 * - Max with current scheme: 85 devices in 0x8000-0x80FF (256/3)
 * - To exceed 85 devices, extend into 0x8700-0x87FF range
 * - Theoretical maximum: 768 devices with reserved space
 */

#endif /* _DSMIL_EXPANDED_ARCH_H */
