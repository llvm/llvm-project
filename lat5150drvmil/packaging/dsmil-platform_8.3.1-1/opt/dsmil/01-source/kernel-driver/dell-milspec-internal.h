/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Dell Military Specification Internal Definitions
 */

#ifndef _DELL_MILSPEC_INTERNAL_H
#define _DELL_MILSPEC_INTERNAL_H

#include <linux/types.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/dell-smbios.h>
#include <linux/ring_buffer.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>
#include <linux/gpio/consumer.h>
#include <linux/i2c.h>
#include <linux/tpm.h>
#include <linux/workqueue.h>

/* Debug Levels */
#define MILSPEC_DBG_INIT    0x0001
#define MILSPEC_DBG_ACPI    0x0002
#define MILSPEC_DBG_SMBIOS  0x0004
#define MILSPEC_DBG_WMI     0x0008
#define MILSPEC_DBG_EVENT   0x0010
#define MILSPEC_DBG_SECURITY 0x0020
#define MILSPEC_DBG_CRYPTO  0x0040
#define MILSPEC_DBG_GPIO    0x0080

extern unsigned int milspec_debug;

#define milspec_dbg(level, fmt, ...) \
do { \
    if (milspec_debug & level) \
        pr_debug("MIL-SPEC: " fmt, ##__VA_ARGS__); \
} while (0)

/* Hardware Addresses - Discovered Values */
#define DELL_MILSPEC_MMIO_BASE     0xFED40000
#define DELL_MILSPEC_MMIO_SIZE     0x1000
#define DELL_GPIO_COMMUNITY1_BASE  0xFD6E0000
#define DELL_GPIO_COMMUNITY1_SIZE  0x10000

/* Test Point GPIO Numbers */
#define GPIO_TP_MODE5         147
#define GPIO_TP_PARANOID      148
#define GPIO_J_SERVICE_PIN5   245
#define GPIO_J_SERVICE_PIN6   246
#define GPIO_CHASSIS_INTRUSION 384
#define GPIO_TAMPER_DETECT    385

/* Register Offsets */
#define MILSPEC_REG_STATUS    0x00
#define MILSPEC_REG_CONTROL   0x04
#define MILSPEC_REG_MODE5     0x08
#define MILSPEC_REG_DSMIL     0x0C
#define MILSPEC_REG_FEATURES  0x10
#define MILSPEC_REG_ACTIVATION 0x20

/* Boot Progress Stages */
#define BOOT_STAGE_EARLY    0x01
#define BOOT_STAGE_ACPI     0x02
#define BOOT_STAGE_SMBIOS   0x04
#define BOOT_STAGE_WMI      0x08
#define BOOT_STAGE_GPIO     0x10
#define BOOT_STAGE_CRYPTO   0x20
#define BOOT_STAGE_COMPLETE 0x40

/* WMI GUIDs */
#define DELL_MILSPEC_EVENT_GUID "85C8A4F9-5A9B-4B6A-B180-92F83AE6B5C3"
#define DELL_MILSPEC_METHOD_GUID "A80593CE-A997-11DA-B012-B622A1EF5492"

/* ACPI Methods */
#define MILSPEC_ACPI_INIT     "_INI"
#define MILSPEC_ACPI_STATUS   "_STA"
#define MILSPEC_ACPI_ENABLE   "ENBL"
#define MILSPEC_ACPI_DISABLE  "DSBL"
#define MILSPEC_ACPI_WIPE     "WIPE"
#define MILSPEC_ACPI_QUERY    "QURY"

/* ATECC608B Crypto Chip */
#define ATECC608B_I2C_ADDR        0x60
#define ATECC608B_WAKE_DELAY_US   1500

/* TME (Total Memory Encryption) MSRs */
#define MSR_IA32_TME_ACTIVATE     0x982
#define TME_ACTIVATE_ENABLED      BIT(0)

/* Log buffer size */
#define LOG_BUFFER_SIZE           65536

/* Global State Structure */
struct milspec_state {
    bool mode5_enabled;
    int mode5_level;
    bool dsmil_active[12];
    int dsmil_mode;
    bool service_mode;
    bool intrusion_detected;
    bool emergency_wipe_armed;
    ktime_t activation_time;
    u32 activation_count;
    u32 error_count;
    spinlock_t lock;
};

/* Internal Device Structure */
struct milspec_device {
    struct platform_device *pdev;
    void __iomem *mmio_base;
    struct mutex lock;

    /* State */
    struct milspec_state *state;

    /* Event logging */
    struct ring_buffer *event_buffer;

    /* ACPI handles */
    struct acpi_device *acpi_dev;
    struct acpi_device *dsmil_devices[12];

    /* WMI */
    struct wmi_device *wmi_dev;

    /* GPIO */
    struct gpio_desc *gpio_mode5;
    struct gpio_desc *gpio_paranoid;
    struct gpio_desc *gpio_intrusion;
    struct gpio_desc *gpio_tamper;

    /* TPM */
    struct tpm_chip *tpm;

    /* Crypto */
    struct i2c_client *crypto_client;

    /* Timers and work */
    struct delayed_work monitor_work;
    struct delayed_work intrusion_work;
    struct timer_list watchdog_timer;

    /* Character device */
    struct cdev cdev;
    dev_t devno;
    struct class *class;

    /* Statistics */
    atomic_t activation_count;
    atomic_t error_count;
};

/* Function Prototypes */
int milspec_acpi_init(struct milspec_device *mdev);
int milspec_smbios_init(struct milspec_device *mdev);
int milspec_wmi_init(struct milspec_device *mdev);
int milspec_gpio_init(struct milspec_device *mdev);
int milspec_security_init(struct milspec_device *mdev);
int milspec_crypto_init(struct milspec_device *mdev);

void milspec_acpi_cleanup(struct milspec_device *mdev);
void milspec_smbios_cleanup(struct milspec_device *mdev);
void milspec_wmi_cleanup(struct milspec_device *mdev);
void milspec_gpio_cleanup(struct milspec_device *mdev);
void milspec_security_cleanup(struct milspec_device *mdev);
void milspec_crypto_cleanup(struct milspec_device *mdev);

/* Event logging */
void milspec_log_event(struct milspec_device *mdev, u32 type,
                       u32 data1, u32 data2, const char *msg);

/* Hardware control */
int milspec_force_activate(struct milspec_device *mdev);
int milspec_set_mode5_level(struct milspec_device *mdev, int level);
int milspec_activate_dsmil(struct milspec_device *mdev, int mode);

#endif /* _DELL_MILSPEC_INTERNAL_H */
