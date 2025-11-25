/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Dell Military Specification Hardware Registers
 */

#ifndef _DELL_MILSPEC_REGS_H
#define _DELL_MILSPEC_REGS_H

/* Status Register Bits */
#define MILSPEC_STATUS_READY        BIT(0)
#define MILSPEC_STATUS_MODE5_ACTIVE BIT(1)
#define MILSPEC_STATUS_DSMIL_ACTIVE BIT(2)
#define MILSPEC_STATUS_SERVICE_MODE BIT(3)
#define MILSPEC_STATUS_LOCKED       BIT(4)
#define MILSPEC_STATUS_INTRUSION    BIT(5)
#define MILSPEC_STATUS_CRYPTO_READY BIT(6)
#define MILSPEC_STATUS_TPM_OK       BIT(7)
#define MILSPEC_STATUS_ERROR        BIT(31)

/* Control Register Bits */
#define MILSPEC_CTRL_ENABLE         BIT(0)
#define MILSPEC_CTRL_RESET          BIT(1)
#define MILSPEC_CTRL_FORCE_ACTIVATE BIT(2)
#define MILSPEC_CTRL_SERVICE_MODE   BIT(3)
#define MILSPEC_CTRL_CLEAR_EVENTS   BIT(4)
#define MILSPEC_CTRL_EMERGENCY_WIPE BIT(7)
#define MILSPEC_CTRL_UNLOCK_MASK    0x4D494C53 /* 'MILS' */

/* Mode 5 Register */
#define MODE5_LEVEL_MASK            0x07
#define MODE5_VM_MIGRATION          BIT(3)
#define MODE5_INTRUSION_WIPE        BIT(4)
#define MODE5_VENDOR_OVERRIDE       BIT(5)
#define MODE5_TPM_BIND              BIT(6)
#define MODE5_PERMANENT             BIT(7)

/* DSMIL Register */
#define DSMIL_DEVICE_MASK           0xFFF /* 12 devices */
#define DSMIL_MODE_SHIFT            16
#define DSMIL_MODE_MASK             (0x3 << DSMIL_MODE_SHIFT)
#define DSMIL_ENHANCED_CRYPTO       BIT(20)
#define DSMIL_TACTICAL_COMMS        BIT(21)
#define DSMIL_SECURE_GPS            BIT(22)
#define DSMIL_EMERGENCY_BEACON      BIT(23)
#define DSMIL_DATA_AT_REST          BIT(24)
#define DSMIL_SECURE_BOOT           BIT(25)

/* Feature Register */
#define FEATURE_TPM_ATTESTATION     BIT(0)
#define FEATURE_HARDWARE_CRYPTO     BIT(1)
#define FEATURE_SECURE_WIPE         BIT(2)
#define FEATURE_INTRUSION_DETECT    BIT(3)
#define FEATURE_ENCRYPTED_MEMORY    BIT(4)
#define FEATURE_DMA_PROTECTION      BIT(5)
#define FEATURE_JTAG_LOCK           BIT(6)
#define FEATURE_TEMPEST_MODE        BIT(7)
#define FEATURE_TME_ENABLED         BIT(8)
#define FEATURE_TME_CAPABILITY      BIT(9)
#define FEATURE_NPU_SECURITY        BIT(10)
#define FEATURE_SECURE_ENCLAVE      BIT(11)

/* Activation Register */
#define ACTIVATION_MAGIC            0x41435456 /* 'ACTV' */
#define ACTIVATION_IN_PROGRESS      BIT(0)
#define ACTIVATION_COMPLETE         BIT(1)
#define ACTIVATION_ERROR            BIT(2)
#define ACTIVATION_TIMEOUT          BIT(3)

/* Dell SMBIOS Token Ranges */
#define DELL_MILSPEC_TOKEN_BASE     0x8000
#define DELL_MILSPEC_TOKEN_MAX      0x8014

/* Specific Tokens */
#define TOKEN_MODE5_ENABLE          0x8000
#define TOKEN_MODE5_LEVEL           0x8001
#define TOKEN_DSMIL_ENABLE          0x8002
#define TOKEN_DSMIL_MODE            0x8003
#define TOKEN_SERVICE_MODE          0x8004
#define TOKEN_JRTC1_CONFIG          0x8005
#define TOKEN_SECURE_WIPE           0x8006
#define TOKEN_INTRUSION_ACTION      0x8007
#define TOKEN_CRYPTO_ENABLE         0x8008
#define TOKEN_TPM_POLICY            0x8009
#define TOKEN_TEMPEST_MODE          0x800A
#define TOKEN_JTAG_CONTROL          0x800B
#define TOKEN_FIRMWARE_LOCK         0x800C
#define TOKEN_VENDOR_OVERRIDE       0x800D
#define TOKEN_EMERGENCY_ACCESS      0x800E
#define TOKEN_AUDIT_CONFIG          0x800F
#define TOKEN_NETWORK_ISOLATION     0x8010
#define TOKEN_USB_POLICY            0x8011
#define TOKEN_DISPLAY_SECURITY      0x8012
#define TOKEN_ACOUSTIC_SECURITY     0x8013
#define TOKEN_DEPLOYMENT_MODE       0x8014

/* Hardware Test Points */
#define TP_MODE5_ENABLE             0x01
#define TP_PARANOID_MODE            0x02
#define TP_SERVICE_JUMPER           0x04
#define TP_JTAG_ENABLE              0x08

/* Security Policy Bits */
#define POLICY_ENFORCE_SECUREBOOT   BIT(0)
#define POLICY_REQUIRE_TPM          BIT(1)
#define POLICY_BLOCK_USB_STORAGE    BIT(2)
#define POLICY_DISABLE_CAMERA       BIT(3)
#define POLICY_DISABLE_MICROPHONE   BIT(4)
#define POLICY_DISABLE_BLUETOOTH    BIT(5)
#define POLICY_DISABLE_WIFI         BIT(6)
#define POLICY_AUDIT_ALL_ACCESS     BIT(7)

#endif /* _DELL_MILSPEC_REGS_H */
