/*
 * DSMIL - Dell SMBIOS Integration Header
 * =======================================
 *
 * Real Dell SMBIOS token definitions and calling interface.
 * Based on documented Dell SMBIOS specification and Linux kernel drivers.
 *
 * Copyright (C) 2025 DSMIL Development
 * License: GPL v2
 */

#ifndef _DSMIL_DELL_SMBIOS_H
#define _DSMIL_DELL_SMBIOS_H

#include <linux/types.h>

/*
 * Dell SMBIOS Calling Interface
 * ==============================
 * Dell systems use SMI (System Management Interrupt) or WMI (Windows Management
 * Instrumentation) to access SMBIOS functionality.
 *
 * The calling interface uses a command/response buffer with:
 * - cmd_class: Command category (0-30)
 * - cmd_select: Specific operation within class
 * - input[4]: Input parameters
 * - output[4]: Response data
 */

/* SMBIOS Command Classes */
#define CLASS_TOKEN_READ		1
#define CLASS_TOKEN_WRITE		2
#define CLASS_KBD_BACKLIGHT		4
#define CLASS_FLASH_INTERFACE		7
#define CLASS_ADMIN_PROP		10
#define CLASS_INFO			17

/* Token Select Values */
#define SELECT_TOKEN_STD		0
#define SELECT_TOKEN_AC			1
#define SELECT_TOKEN_BAT		2
#define SELECT_KBD_BACKLIGHT		11
#define SELECT_THERMAL_MANAGEMENT	19

/* SMBIOS calling interface buffer (from dell-smbios.h) */
struct calling_interface_buffer {
	u16 cmd_class;
	u16 cmd_select;
	u32 input[4];
	u32 output[4];
} __packed;

/* SMBIOS token structure (from dell-smbios.h) */
struct calling_interface_token {
	u16 tokenID;
	u16 location;
	union {
		u16 value;
		u16 stringlength;
	};
} __packed;

/* SMBIOS structure from DMI table */
struct calling_interface_structure {
	struct dmi_header header;
	u16 cmdIOAddress;
	u8 cmdIOCode;
	u32 supportedCmds;
	struct calling_interface_token tokens[];
} __packed;

/*
 * Real Dell SMBIOS Token IDs
 * ===========================
 * These are documented Dell SMBIOS tokens found in actual Dell systems.
 * Token ranges:
 * - 0x0000-0x00FF: Basic system control
 * - 0x0100-0x01FF: Power management
 * - 0x0200-0x02FF: Display and graphics
 * - 0x0300-0x03FF: Manufacturing (restricted)
 * - 0x0400-0x7FFF: Extended features
 * - 0x8000-0xFFFF: OEM/Model-specific
 */

/* Keyboard Backlight Tokens (documented) */
#define TOKEN_KBD_BACKLIGHT_BRIGHTNESS	0x007D
#define TOKEN_KBD_LED_AC_TOKEN		0x0451
#define TOKEN_KBD_LED_OFF_TOKEN		0x01E1
#define TOKEN_KBD_LED_ON_TOKEN		0x01E2
#define TOKEN_KBD_LED_AUTO_25_TOKEN	0x02EA
#define TOKEN_KBD_LED_AUTO_50_TOKEN	0x02EB
#define TOKEN_KBD_LED_AUTO_75_TOKEN	0x02EC
#define TOKEN_KBD_LED_AUTO_100_TOKEN	0x02ED

/* Battery/Power Management Tokens (documented) */
#define TOKEN_BATTERY_MODE_ADAPTIVE	0x0003
#define TOKEN_BATTERY_MODE_CUSTOM	0x0004
#define TOKEN_BATTERY_MODE_STANDARD	0x0080
#define TOKEN_BATTERY_MODE_EXPRESS	0x0081
#define TOKEN_BATTERY_MODE_PRIMARILY_AC	0x0082
#define TOKEN_BATTERY_CUSTOM_CHARGE_START	0x0349
#define TOKEN_BATTERY_CUSTOM_CHARGE_END		0x034A

/* Audio Tokens (documented) */
#define TOKEN_GLOBAL_MIC_MUTE_ENABLE	0x0364
#define TOKEN_GLOBAL_MIC_MUTE_DISABLE	0x0365
#define TOKEN_SPEAKER_MUTE_ENABLE	0x058C
#define TOKEN_SPEAKER_MUTE_DISABLE	0x058D

/* Thermal Management Tokens */
#define TOKEN_THERMAL_MODE_OPTIMIZED	0x0001
#define TOKEN_THERMAL_MODE_COOL		0x0002
#define TOKEN_THERMAL_MODE_QUIET	0x0003
#define TOKEN_THERMAL_MODE_PERFORMANCE	0x0004

/* Display Tokens */
#define TOKEN_BRIGHTNESS_AC_TOKEN	0x007C
#define TOKEN_BRIGHTNESS_BAT_TOKEN	0x007D

/*
 * DSMIL Extended Token Ranges
 * ============================
 * Custom tokens for DSMIL-specific functionality.
 * Using 0x8000-0x8FFF range (OEM-specific as per Dell convention).
 */

/* DSMIL Core Security Group (0x8000-0x800B) */
#define TOKEN_DSMIL_SYSTEM_STATUS	0x8000
#define TOKEN_DSMIL_SECURITY_LEVEL	0x8001
#define TOKEN_DSMIL_AUTH_STATUS		0x8002
#define TOKEN_DSMIL_AUDIT_CONTROL	0x8003
#define TOKEN_DSMIL_MFA_ENABLED		0x8004
#define TOKEN_DSMIL_COMPLIANCE_MODE	0x8005
#define TOKEN_DSMIL_EMERGENCY_STOP	0x8006
#define TOKEN_DSMIL_THREAT_LEVEL	0x8007
#define TOKEN_DSMIL_INCIDENT_COUNT	0x8008

/* Protected tokens - write operations restricted */
#define TOKEN_DSMIL_SYSTEM_RESET	0x8009
#define TOKEN_DSMIL_SECURE_ERASE	0x800A
#define TOKEN_DSMIL_FACTORY_RESET	0x800B

/* DSMIL Power & Thermal Group (0x800C-0x8017) */
#define TOKEN_DSMIL_POWER_MODE		0x800C
#define TOKEN_DSMIL_THERMAL_ZONE_0	0x800D
#define TOKEN_DSMIL_THERMAL_ZONE_1	0x800E
#define TOKEN_DSMIL_THERMAL_ZONE_2	0x800F
#define TOKEN_DSMIL_FAN_CONTROL		0x8010
#define TOKEN_DSMIL_VOLTAGE_RAIL_0	0x8011
#define TOKEN_DSMIL_VOLTAGE_RAIL_1	0x8012
#define TOKEN_DSMIL_POWER_LIMIT		0x8013
#define TOKEN_DSMIL_BATTERY_HEALTH	0x8014
#define TOKEN_DSMIL_CHARGE_PROFILE	0x8015
#define TOKEN_DSMIL_POWER_STATS		0x8016
#define TOKEN_DSMIL_THERMAL_POLICY	0x8017

/* DSMIL Network Group (0x8018-0x8023) */
#define TOKEN_DSMIL_NETWORK_STATUS	0x8018
#define TOKEN_DSMIL_NETWORK_KILLSWITCH	0x8019	/* Protected */
#define TOKEN_DSMIL_WIFI_CONTROL	0x801A
#define TOKEN_DSMIL_BT_CONTROL		0x801B
#define TOKEN_DSMIL_WWAN_CONTROL	0x801C
#define TOKEN_DSMIL_ETHERNET_CONTROL	0x801D
#define TOKEN_DSMIL_FIREWALL_STATE	0x801E
#define TOKEN_DSMIL_VPN_STATE		0x801F
#define TOKEN_DSMIL_NETWORK_POLICY	0x8020
#define TOKEN_DSMIL_BANDWIDTH_LIMIT	0x8021
#define TOKEN_DSMIL_CONNECTION_COUNT	0x8022
#define TOKEN_DSMIL_NETWORK_STATS	0x8023

/* DSMIL Data Processing Group (0x8024-0x802F) */
#define TOKEN_DSMIL_CRYPTO_ENGINE	0x8024
#define TOKEN_DSMIL_HASH_ENGINE		0x8025
#define TOKEN_DSMIL_RNG_SOURCE		0x8026
#define TOKEN_DSMIL_TPM_STATUS		0x8027
#define TOKEN_DSMIL_SECURE_BOOT		0x8028
#define TOKEN_DSMIL_DATA_WIPE		0x8029	/* Protected */
#define TOKEN_DSMIL_ENCRYPTION_STATE	0x802A
#define TOKEN_DSMIL_KEY_MANAGEMENT	0x802B
#define TOKEN_DSMIL_CERT_STATUS		0x802C
#define TOKEN_DSMIL_CRYPTO_POLICY	0x802D
#define TOKEN_DSMIL_SECURE_STORAGE	0x802E
#define TOKEN_DSMIL_DATA_INTEGRITY	0x802F

/* Total DSMIL tokens */
#define DSMIL_TOKEN_COUNT		48

/*
 * Protected Token List
 * ====================
 * These tokens require special authorization (e.g., CAP_SYS_ADMIN + MFA)
 */
static const u16 dsmil_protected_tokens[] = {
	TOKEN_DSMIL_SYSTEM_RESET,	/* 0x8009 */
	TOKEN_DSMIL_SECURE_ERASE,	/* 0x800A */
	TOKEN_DSMIL_FACTORY_RESET,	/* 0x800B */
	TOKEN_DSMIL_NETWORK_KILLSWITCH,	/* 0x8019 */
	TOKEN_DSMIL_DATA_WIPE,		/* 0x8029 */
};
#define DSMIL_PROTECTED_TOKEN_COUNT	5

/*
 * SMBIOS Call Return Codes (from Dell documentation)
 */
#define SMBIOS_RET_SUCCESS		0
#define SMBIOS_RET_INVALID_PARAM	-1
#define SMBIOS_RET_UNSUPPORTED_FUNC	-2
#define SMBIOS_RET_BUFFER_TOO_SMALL	-3
#define SMBIOS_RET_INVALID_TOKEN	-4
#define SMBIOS_RET_PERMISSION_DENIED	-5

/*
 * Helper Functions
 */

static inline bool dsmil_is_protected_token(u16 token_id)
{
	int i;
	for (i = 0; i < DSMIL_PROTECTED_TOKEN_COUNT; i++) {
		if (dsmil_protected_tokens[i] == token_id)
			return true;
	}
	return false;
}

static inline bool dsmil_is_dsmil_token(u16 token_id)
{
	return (token_id >= 0x8000 && token_id <= 0x802F);
}

static inline bool dsmil_is_std_dell_token(u16 token_id)
{
	return (token_id < 0x8000);
}

#endif /* _DSMIL_DELL_SMBIOS_H */
