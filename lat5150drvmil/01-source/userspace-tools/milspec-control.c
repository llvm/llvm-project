/*
 * Dell MIL-SPEC Control Utility
 * Command-line tool for controlling dell-milspec driver features
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <linux/dell-milspec.h>

/* Color codes */
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"

/* Default device path */
#define DEFAULT_DEVICE "/dev/milspec"

/* Mode 5 level strings */
static const char *mode5_level_str[] = {
    "DISABLED",
    "STANDARD",
    "ENHANCED",
    "PARANOID",
    "PARANOID_PLUS"
};

/* DSMIL mode strings */
static const char *dsmil_mode_str[] = {
    "OFF",
    "BASIC",
    "ENHANCED",
    "CLASSIFIED"
};

/* Global config */
struct config {
    const char *device;
    int verbose;
    int force;
    int use_color;
};

/* Print with color if enabled */
static void print_color(struct config *cfg, const char *color, const char *fmt, ...)
{
    va_list args;

    if (cfg->use_color && isatty(STDOUT_FILENO))
        printf("%s", color);

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (cfg->use_color && isatty(STDOUT_FILENO))
        printf("%s", COLOR_RESET);
}

/* Show current status */
static int show_status(struct config *cfg, int fd)
{
    struct milspec_status status;
    struct milspec_security security;

    if (ioctl(fd, MILSPEC_IOC_GET_STATUS, &status) < 0) {
        perror("ioctl(GET_STATUS)");
        return -1;
    }

    print_color(cfg, COLOR_BOLD, "\n=== Dell MIL-SPEC Status ===\n\n");

    printf("API Version:      0x%06x\n", status.api_version);
    printf("Boot Progress:    0x%02x", status.boot_progress);

    if (cfg->verbose) {
        printf(" (");
        if (status.boot_progress & 0x01) printf("EARLY ");
        if (status.boot_progress & 0x02) printf("ACPI ");
        if (status.boot_progress & 0x04) printf("SMBIOS ");
        if (status.boot_progress & 0x08) printf("WMI ");
        if (status.boot_progress & 0x10) printf("GPIO ");
        if (status.boot_progress & 0x20) printf("CRYPTO ");
        if (status.boot_progress & 0x40) printf("COMPLETE");
        printf(")");
    }
    printf("\n");

    printf("Activation Count: %u\n", status.activation_count);
    printf("Error Count:      %u\n", status.error_count);

    if (status.activation_time_ns > 0) {
        printf("Last Activation:  %llu seconds ago\n",
               (unsigned long long)(status.uptime_ns - status.activation_time_ns) / 1000000000);
    }

    print_color(cfg, COLOR_CYAN, "\nMode 5 Security:\n");
    printf("  Status:         %s\n", status.mode5_enabled ?
    print_color(cfg, COLOR_GREEN, "ENABLED") :
    print_color(cfg, COLOR_YELLOW, "DISABLED"));
    printf("  Level:          %s (%d)\n", mode5_level_str[status.mode5_level], status.mode5_level);

    if (status.mode5_level == MODE5_PARANOID_PLUS) {
        print_color(cfg, COLOR_RED, "  WARNING:        IRREVERSIBLE CONFIGURATION!\n");
    }

    print_color(cfg, COLOR_CYAN, "\nDSMIL Subsystems:\n");
    printf("  Mode:           %s (%d)\n", dsmil_mode_str[status.dsmil_mode], status.dsmil_mode);
    printf("  Active Devices: ");

    int count = 0;
    for (int i = 0; i < 10; i++) {
        if (status.dsmil_active[i]) {
            if (count++) printf(", ");
            printf("DSMIL0D%d", i);
        }
    }
    if (count == 0)
        printf("None");
    printf("\n");

    print_color(cfg, COLOR_CYAN, "\nSystem State:\n");
    printf("  Service Mode:   %s\n", status.service_mode ?
    print_color(cfg, COLOR_YELLOW, "ACTIVE") : "Inactive");
    printf("  TPM Measured:   %s\n", status.tpm_measured ? "Yes" : "No");
    printf("  Crypto Present: %s\n", status.crypto_present ?
    print_color(cfg, COLOR_GREEN, "Yes") : "No");

    /* Get security status */
    if (ioctl(fd, MILSPEC_IOC_GET_SECURITY, &security) == 0) {
        print_color(cfg, COLOR_CYAN, "\nSecurity Status:\n");

        if (security.intrusion_detected || security.tamper_detected) {
            print_color(cfg, COLOR_RED, "  *** SECURITY ALERT ***\n");
        }

        printf("  Intrusion:      %s\n", security.intrusion_detected ?
        print_color(cfg, COLOR_RED, "DETECTED!") : "Clear");
        printf("  Tamper:         %s\n", security.tamper_detected ?
        print_color(cfg, COLOR_RED, "DETECTED!") : "Clear");
        printf("  Secure Boot:    %s\n", security.secure_boot_enabled ? "Enabled" : "Disabled");
        printf("  Measured Boot:  %s\n", security.measured_boot ? "Yes" : "No");
        printf("  Memory Encrypt: %s\n", security.encrypted_memory ? "Active" : "Inactive");
        printf("  DMA Protection: %s\n", security.dma_protection ? "Enabled" : "Disabled");

        if (cfg->verbose && security.crypto_serial[0]) {
            printf("  Crypto Serial:  ");
            for (int i = 0; i < 9; i++)
                printf("%02X", security.crypto_serial[i]);
            printf("\n");
        }
    }

    printf("\n");
    return 0;
}

/* Set Mode 5 level */
static int set_mode5(struct config *cfg, int fd, int level)
{
    if (level < 0 || level > 4) {
        fprintf(stderr, "Invalid Mode 5 level: %d (valid: 0-4)\n", level);
        return -1;
    }

    if (level == MODE5_PARANOID_PLUS && !cfg->force) {
        print_color(cfg, COLOR_RED,
                    "\nWARNING: Mode 5 PARANOID_PLUS is IRREVERSIBLE!\n");
        printf("This will permanently lock your system configuration.\n");
        printf("Use --force to confirm this action.\n\n");
        return -1;
    }

    printf("Setting Mode 5 to %s (%d)...\n", mode5_level_str[level], level);

    if (ioctl(fd, MILSPEC_IOC_SET_MODE5, &level) < 0) {
        perror("ioctl(SET_MODE5)");
        return -1;
    }

    print_color(cfg, COLOR_GREEN, "Mode 5 level set successfully.\n");
    return 0;
}

/* Activate DSMIL subsystems */
static int activate_dsmil(struct config *cfg, int fd, int mode)
{
    if (mode < 0 || mode > 3) {
        fprintf(stderr, "Invalid DSMIL mode: %d (valid: 0-3)\n", mode);
        return -1;
    }

    printf("Activating DSMIL subsystems in %s mode (%d)...\n",
           dsmil_mode_str[mode], mode);

    if (ioctl(fd, MILSPEC_IOC_ACTIVATE_DSMIL, &mode) < 0) {
        perror("ioctl(ACTIVATE_DSMIL)");
        return -1;
    }

    print_color(cfg, COLOR_GREEN, "DSMIL subsystems activated.\n");
    return 0;
}

/* Force activation of all features */
static int force_activate(struct config *cfg, int fd)
{
    printf("Forcing activation of all MIL-SPEC features...\n");

    if (ioctl(fd, MILSPEC_IOC_FORCE_ACTIVATE, NULL) < 0) {
        perror("ioctl(FORCE_ACTIVATE)");
        return -1;
    }

    print_color(cfg, COLOR_GREEN, "All features activated.\n");
    return 0;
}

/* Trigger TPM measurement */
static int tpm_measure(struct config *cfg, int fd)
{
    printf("Triggering TPM measurement of MIL-SPEC state...\n");

    if (ioctl(fd, MILSPEC_IOC_TPM_MEASURE, NULL) < 0) {
        perror("ioctl(TPM_MEASURE)");
        return -1;
    }

    print_color(cfg, COLOR_GREEN, "TPM measurement completed.\n");
    return 0;
}

/* Emergency wipe */
static int emergency_wipe(struct config *cfg, int fd)
{
    char confirm[64];

    print_color(cfg, COLOR_RED,
                "\n*** EMERGENCY DATA DESTRUCTION ***\n");
    printf("This will IMMEDIATELY and IRREVERSIBLY destroy all data!\n");
    printf("The system will be wiped and rebooted.\n\n");
    printf("Type 'DESTROY ALL DATA' to confirm: ");

    if (fgets(confirm, sizeof(confirm), stdin) == NULL)
        return -1;

    /* Remove newline */
    confirm[strcspn(confirm, "\n")] = '\0';

    if (strcmp(confirm, "DESTROY ALL DATA") != 0) {
        printf("Confirmation failed. Aborting.\n");
        return -1;
    }

    printf("\nInitiating emergency wipe...\n");

    __u32 wipe_code = MILSPEC_WIPE_CONFIRM;
    if (ioctl(fd, MILSPEC_IOC_EMERGENCY_WIPE, &wipe_code) < 0) {
        perror("ioctl(EMERGENCY_WIPE)");
        return -1;
    }

    /* Should not reach here if wipe succeeded */
    return 0;
}

/* Show help */
static void usage(const char *prog)
{
    printf("Usage: %s [options] [command]\n", prog);
    printf("\nCommands:\n");
    printf("  status                    Show current MIL-SPEC status (default)\n");
    printf("  mode5 <level>             Set Mode 5 security level (0-4)\n");
    printf("  dsmil <mode>              Activate DSMIL subsystems (0-3)\n");
    printf("  activate                  Force activation of all features\n");
    printf("  measure                   Trigger TPM measurement\n");
    printf("  wipe                      Emergency data destruction\n");
    printf("\nOptions:\n");
    printf("  -d, --device PATH         Device path (default: %s)\n", DEFAULT_DEVICE);
    printf("  -f, --force               Force dangerous operations\n");
    printf("  -c, --color               Use color output\n");
    printf("  -v, --verbose             Verbose output\n");
    printf("  -h, --help                Show this help\n");
    printf("\nMode 5 Levels:\n");
    printf("  0 - DISABLED              No Mode 5 protection\n");
    printf("  1 - STANDARD              Basic protection, VM migration allowed\n");
    printf("  2 - ENHANCED              VMs locked to hardware\n");
    printf("  3 - PARANOID              Secure wipe on intrusion\n");
    printf("  4 - PARANOID_PLUS         Maximum security (IRREVERSIBLE!)\n");
    printf("\nDSMIL Modes:\n");
    printf("  0 - OFF                   All subsystems disabled\n");
    printf("  1 - BASIC                 Essential features only\n");
    printf("  2 - ENHANCED              Full tactical capabilities\n");
    printf("  3 - CLASSIFIED            Restricted mode\n");
    printf("\nExamples:\n");
    printf("  %s status                 Show current status\n", prog);
    printf("  %s mode5 2                Set Mode 5 to ENHANCED\n", prog);
    printf("  %s dsmil 1                Activate DSMIL in BASIC mode\n", prog);
}

int main(int argc, char *argv[])
{
    struct config cfg = {
        .device = DEFAULT_DEVICE,
        .verbose = 0,
        .force = 0,
        .use_color = 1
    };

    int opt, option_index = 0;
    int fd, ret = 0;

    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'},
        {"force", no_argument, 0, 'f'},
        {"color", no_argument, 0, 'c'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    /* Parse options */
    while ((opt = getopt_long(argc, argv, "d:fcvh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd':
                cfg.device = optarg;
                break;
            case 'f':
                cfg.force = 1;
                break;
            case 'c':
                cfg.use_color = 1;
                break;
            case 'v':
                cfg.verbose = 1;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    /* Open device */
    fd = open(cfg.device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Cannot open %s: %s\n", cfg.device, strerror(errno));
        fprintf(stderr, "Is the dell-milspec driver loaded?\n");
        return 1;
    }

    /* Process command */
    if (optind >= argc) {
        /* No command specified, show status */
        ret = show_status(&cfg, fd);
    } else {
        const char *cmd = argv[optind];

        if (strcmp(cmd, "status") == 0) {
            ret = show_status(&cfg, fd);
        }
        else if (strcmp(cmd, "mode5") == 0) {
            if (optind + 1 >= argc) {
                fprintf(stderr, "mode5 requires a level argument\n");
                ret = -1;
            } else {
                int level = atoi(argv[optind + 1]);
                ret = set_mode5(&cfg, fd, level);
            }
        }
        else if (strcmp(cmd, "dsmil") == 0) {
            if (optind + 1 >= argc) {
                fprintf(stderr, "dsmil requires a mode argument\n");
                ret = -1;
            } else {
                int mode = atoi(argv[optind + 1]);
                ret = activate_dsmil(&cfg, fd, mode);
            }
        }
        else if (strcmp(cmd, "activate") == 0) {
            ret = force_activate(&cfg, fd);
        }
        else if (strcmp(cmd, "measure") == 0) {
            ret = tpm_measure(&cfg, fd);
        }
        else if (strcmp(cmd, "wipe") == 0) {
            ret = emergency_wipe(&cfg, fd);
        }
        else {
            fprintf(stderr, "Unknown command: %s\n", cmd);
            usage(argv[0]);
            ret = -1;
        }
    }

    close(fd);
    return ret ? 1 : 0;
}
