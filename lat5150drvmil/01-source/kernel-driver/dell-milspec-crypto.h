/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Dell Military Specification Cryptographic Hardware Support
 */

#ifndef _DELL_MILSPEC_CRYPTO_H
#define _DELL_MILSPEC_CRYPTO_H

#include <linux/types.h>
#include <linux/i2c.h>

/* ATECC608B I2C Configuration */
#define ATECC608B_I2C_ADDR          0x60
#define ATECC608B_WAKE_DELAY_US     1500
#define ATECC608B_EXEC_TIME_MS      50
#define ATECC608B_MAX_RETRIES       3

/* ATECC608B Commands */
#define ATECC_CMD_INFO              0x30
#define ATECC_CMD_GENKEY            0x40
#define ATECC_CMD_SIGN              0x41
#define ATECC_CMD_VERIFY            0x45
#define ATECC_CMD_WRITE             0x12
#define ATECC_CMD_READ              0x02
#define ATECC_CMD_LOCK              0x17
#define ATECC_CMD_RANDOM            0x1B
#define ATECC_CMD_DERIVEKEY         0x1C
#define ATECC_CMD_UPDATEEXTRA       0x20
#define ATECC_CMD_COUNTER           0x24
#define ATECC_CMD_ECDH             0x43
#define ATECC_CMD_MAC               0x08
#define ATECC_CMD_SHA               0x47
#define ATECC_CMD_WIPE              0xFF  /* Custom for emergency wipe */

/* ATECC608B Zones */
#define ATECC_ZONE_CONFIG           0x00
#define ATECC_ZONE_OTP              0x01
#define ATECC_ZONE_DATA             0x02

/* ATECC608B Packet Structure */
struct atecc_packet {
    u8 function;
    u8 word_addr;
    u8 count;
    u8 *data;
    u16 crc;
} __packed;

/* ATECC608B Response Status Codes */
#define ATECC_STATUS_SUCCESS        0x00
#define ATECC_STATUS_CHECKMAC_FAIL  0x01
#define ATECC_STATUS_PARSE_ERROR    0x03
#define ATECC_STATUS_ECC_FAULT      0x05
#define ATECC_STATUS_EXEC_ERROR     0x0F
#define ATECC_STATUS_WAKE_OK        0x11
#define ATECC_STATUS_COMM_ERROR     0xFF

/* Crypto chip states */
#define CRYPTO_STATE_SLEEP          0x00
#define CRYPTO_STATE_IDLE           0x01
#define CRYPTO_STATE_ACTIVE         0x02
#define CRYPTO_STATE_ERROR          0x03

/* Key Slot Assignments */
#define SLOT_PRIMARY_KEY            0
#define SLOT_ATTESTATION_KEY        1
#define SLOT_ENCRYPTION_KEY         2
#define SLOT_AUTHENTICATION_KEY     3
#define SLOT_SECURE_BOOT_KEY        4
#define SLOT_EMERGENCY_KEY          5
#define SLOT_COMM_KEY               6
#define SLOT_STORAGE_KEY            7
#define SLOT_TPM_BIND_KEY           8
#define SLOT_AUDIT_KEY              9
#define SLOT_RECOVERY_KEY           10
#define SLOT_RESERVED_START         11
#define SLOT_RESERVED_END           15

/* Hardware Security Module Data */
struct atecc608b_data {
    struct i2c_client *client;
    bool present;
    u8 serial[9];
    u8 revision[4];
    u8 state;
    bool locked;
    u32 features;
};

/* Crypto Operations */
struct milspec_crypto_op {
    u8 operation;
    u8 key_slot;
    size_t data_len;
    u8 *data_in;
    u8 *data_out;
    u8 *signature;
    u32 flags;
};

/* Crypto operation types */
#define CRYPTO_OP_ENCRYPT           0x01
#define CRYPTO_OP_DECRYPT           0x02
#define CRYPTO_OP_SIGN              0x03
#define CRYPTO_OP_VERIFY            0x04
#define CRYPTO_OP_DERIVE            0x05
#define CRYPTO_OP_RANDOM            0x06
#define CRYPTO_OP_ATTEST            0x07

/* Crypto operation flags */
#define CRYPTO_FLAG_HARDWARE        BIT(0)
#define CRYPTO_FLAG_TPM_BIND        BIT(1)
#define CRYPTO_FLAG_AUDIT           BIT(2)
#define CRYPTO_FLAG_EMERGENCY       BIT(3)

/* TPM Integration */
#define TPM_PCR_MILSPEC_BASE        16
#define TPM_PCR_MODE5_CONFIG        16
#define TPM_PCR_DSMIL_STATE         17
#define TPM_PCR_CRYPTO_CONFIG       18
#define TPM_PCR_AUDIT_LOG           19

/* Function prototypes for crypto operations */
int milspec_crypto_init(struct atecc608b_data *crypto);
int milspec_crypto_wakeup(struct atecc608b_data *crypto);
int milspec_crypto_sleep(struct atecc608b_data *crypto);
int milspec_crypto_get_info(struct atecc608b_data *crypto);
int milspec_crypto_generate_key(struct atecc608b_data *crypto, u8 slot);
int milspec_crypto_sign(struct atecc608b_data *crypto,
                       u8 slot, u8 *data, size_t len, u8 *signature);
int milspec_crypto_verify(struct atecc608b_data *crypto,
                         u8 slot, u8 *data, size_t len, u8 *signature);
int milspec_crypto_random(struct atecc608b_data *crypto, u8 *buffer, size_t len);
int milspec_crypto_wipe(struct atecc608b_data *crypto);

/* CRC calculation for ATECC608B */
u16 milspec_crypto_crc16(u8 *data, size_t len);

#endif /* _DELL_MILSPEC_CRYPTO_H */
