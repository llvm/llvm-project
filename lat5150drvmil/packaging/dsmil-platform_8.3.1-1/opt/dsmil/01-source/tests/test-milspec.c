/*
 * Dell MIL-SPEC Driver Test Program
 * 
 * Simple test utility to validate IOCTL interface
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#include "dell-milspec.h"

void print_status(struct milspec_status *status) {
    printf("\n=== Dell MIL-SPEC Status ===\n");
    printf("API Version: 0x%06x\n", status->api_version);
    printf("Mode5 Enabled: %s\n", status->mode5_enabled ? "Yes" : "No");
    printf("Mode5 Level: %u (%s)\n", status->mode5_level,
           status->mode5_level == 0 ? "Disabled" :
           status->mode5_level == 1 ? "Standard" :
           status->mode5_level == 2 ? "Enhanced" :
           status->mode5_level == 3 ? "Paranoid" :
           status->mode5_level == 4 ? "Paranoid+" : "Unknown");
    printf("DSMIL Mode: %u (%s)\n", status->dsmil_mode,
           status->dsmil_mode == 0 ? "Off" :
           status->dsmil_mode == 1 ? "Basic" :
           status->dsmil_mode == 2 ? "Enhanced" :
           status->dsmil_mode == 3 ? "Classified" : "Unknown");
    printf("Service Mode: %s\n", status->service_mode ? "Active" : "Inactive");
    
    printf("\nDSMIL Devices:\n");
    for (int i = 0; i < 10; i++) {
        if (status->dsmil_active[i]) {
            printf("  DSMIL0D%d: Active\n", i);
        }
    }
    
    printf("\nSecurity:\n");
    printf("  Intrusion Detected: %s\n", status->intrusion_detected ? "YES!" : "No");
    printf("  TPM Measured: %s\n", status->tpm_measured ? "Yes" : "No");
    printf("  Crypto Chip: %s\n", status->crypto_present ? "Present" : "Not installed (optional)");
    
    printf("\nCounters:\n");
    printf("  Activation Count: %u\n", status->activation_count);
    printf("  Error Count: %u\n", status->error_count);
    printf("  Event Count: %u\n", status->event_count);
    
    if (status->activation_time_ns > 0) {
        time_t activation_time = status->activation_time_ns / 1000000000;
        printf("  Last Activation: %s", ctime(&activation_time));
    }
    
    printf("  Boot Progress: 0x%02x\n", status->boot_progress);
    printf("============================\n\n");
}

void print_events(struct milspec_events *events) {
    printf("\n=== Recent Events ===\n");
    printf("Event Count: %u\n", events->count);
    printf("Lost Events: %u\n", events->lost);
    
    for (int i = 0; i < events->count && i < 64; i++) {
        time_t event_time = events->events[i].timestamp_ns / 1000000000;
        printf("\nEvent %d:\n", i + 1);
        printf("  Type: 0x%08x (", events->events[i].event_type);
        
        switch (events->events[i].event_type) {
            case MILSPEC_EVENT_BOOT: printf("Boot"); break;
            case MILSPEC_EVENT_ACTIVATION: printf("Activation"); break;
            case MILSPEC_EVENT_MODE_CHANGE: printf("Mode Change"); break;
            case MILSPEC_EVENT_ERROR: printf("Error"); break;
            case MILSPEC_EVENT_USER_REQUEST: printf("User Request"); break;
            case MILSPEC_EVENT_SECURITY: printf("Security"); break;
            case MILSPEC_EVENT_POWER: printf("Power"); break;
            case MILSPEC_EVENT_FIRMWARE: printf("Firmware"); break;
            case MILSPEC_EVENT_INTRUSION: printf("Intrusion"); break;
            case MILSPEC_EVENT_CRYPTO: printf("Crypto"); break;
            default: printf("Unknown"); break;
        }
        printf(")\n");
        
        printf("  Data1: 0x%08x\n", events->events[i].data1);
        printf("  Data2: 0x%08x\n", events->events[i].data2);
        if (strlen(events->events[i].message) > 0) {
            printf("  Message: %s\n", events->events[i].message);
        }
        printf("  Time: %s", ctime(&event_time));
    }
    printf("====================\n\n");
}

int test_get_status(int fd) {
    struct milspec_status status;
    
    printf("Testing MILSPEC_IOC_GET_STATUS...\n");
    if (ioctl(fd, MILSPEC_IOC_GET_STATUS, &status) < 0) {
        printf("  FAILED: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  SUCCESS\n");
    print_status(&status);
    return 0;
}

int test_get_events(int fd) {
    struct milspec_events events;
    
    printf("Testing MILSPEC_IOC_GET_EVENTS...\n");
    memset(&events, 0, sizeof(events));
    if (ioctl(fd, MILSPEC_IOC_GET_EVENTS, &events) < 0) {
        printf("  FAILED: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  SUCCESS\n");
    print_events(&events);
    return 0;
}

int test_set_mode5(int fd, int level) {
    printf("Testing MILSPEC_IOC_SET_MODE5 with level %d...\n", level);
    if (ioctl(fd, MILSPEC_IOC_SET_MODE5, &level) < 0) {
        printf("  FAILED: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  SUCCESS\n");
    return 0;
}

int test_activate_dsmil(int fd, int mode) {
    printf("Testing MILSPEC_IOC_ACTIVATE_DSMIL with mode %d...\n", mode);
    if (ioctl(fd, MILSPEC_IOC_ACTIVATE_DSMIL, &mode) < 0) {
        printf("  FAILED: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  SUCCESS\n");
    return 0;
}

int test_force_activate(int fd) {
    printf("Testing MILSPEC_IOC_FORCE_ACTIVATE...\n");
    if (ioctl(fd, MILSPEC_IOC_FORCE_ACTIVATE, NULL) < 0) {
        printf("  FAILED: %s\n", strerror(errno));
        return -1;
    }
    
    printf("  SUCCESS\n");
    return 0;
}

int test_emergency_wipe(int fd, int confirm) {
    __u32 confirm_code = confirm ? MILSPEC_WIPE_CONFIRM : 0;
    
    printf("Testing MILSPEC_IOC_EMERGENCY_WIPE %s...\n", 
           confirm ? "WITH CONFIRMATION" : "without confirmation");
    if (ioctl(fd, MILSPEC_IOC_EMERGENCY_WIPE, &confirm_code) < 0) {
        printf("  FAILED (expected): %s\n", strerror(errno));
        return 0; // Expected to fail without confirmation
    }
    
    printf("  SUCCESS (WARNING: THIS WOULD WIPE DATA!)\n");
    return 0;
}

void print_help(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -h          Show this help\n");
    printf("  -s          Get status only\n");
    printf("  -e          Get events only\n");
    printf("  -m <level>  Set Mode5 level (0-4)\n");
    printf("  -d <mode>   Activate DSMIL mode (0-3)\n");
    printf("  -f          Force activate\n");
    printf("  -w          Test emergency wipe (no confirmation)\n");
    printf("  -W          Test emergency wipe WITH CONFIRMATION (DANGEROUS!)\n");
    printf("  -a          Run all tests (except wipe with confirmation)\n");
    printf("\n");
    printf("Example: %s -a          # Run all safe tests\n", prog);
    printf("         %s -s          # Just get status\n", prog);
    printf("         %s -m 2 -d 1   # Set Mode5=Enhanced, DSMIL=Basic\n", prog);
}

int main(int argc, char *argv[]) {
    int fd;
    int opt;
    int run_all = 0;
    int status_only = 0;
    int events_only = 0;
    int mode5_level = -1;
    int dsmil_mode = -1;
    int force_activate = 0;
    int test_wipe = 0;
    int wipe_confirm = 0;
    
    while ((opt = getopt(argc, argv, "hsem:d:fwWa")) != -1) {
        switch (opt) {
            case 'h':
                print_help(argv[0]);
                return 0;
            case 's':
                status_only = 1;
                break;
            case 'e':
                events_only = 1;
                break;
            case 'm':
                mode5_level = atoi(optarg);
                if (mode5_level < 0 || mode5_level > 4) {
                    fprintf(stderr, "Invalid Mode5 level: %d (must be 0-4)\n", mode5_level);
                    return 1;
                }
                break;
            case 'd':
                dsmil_mode = atoi(optarg);
                if (dsmil_mode < 0 || dsmil_mode > 3) {
                    fprintf(stderr, "Invalid DSMIL mode: %d (must be 0-3)\n", dsmil_mode);
                    return 1;
                }
                break;
            case 'f':
                force_activate = 1;
                break;
            case 'w':
                test_wipe = 1;
                break;
            case 'W':
                test_wipe = 1;
                wipe_confirm = 1;
                fprintf(stderr, "WARNING: Emergency wipe WITH confirmation requested!\n");
                fprintf(stderr, "This is DANGEROUS and could destroy data!\n");
                fprintf(stderr, "Press Ctrl-C now to abort...\n");
                sleep(3);
                break;
            case 'a':
                run_all = 1;
                break;
            default:
                print_help(argv[0]);
                return 1;
        }
    }
    
    printf("Dell MIL-SPEC Driver Test Program\n");
    printf("=================================\n\n");
    
    // Open device
    printf("Opening /dev/milspec...\n");
    fd = open("/dev/milspec", O_RDWR);
    if (fd < 0) {
        perror("Failed to open /dev/milspec");
        printf("\nMake sure:\n");
        printf("1. The dell-milspec module is loaded (lsmod | grep milspec)\n");
        printf("2. The device exists (ls -la /dev/milspec)\n");
        printf("3. You have permission to access it (run as root or check permissions)\n");
        return 1;
    }
    printf("Device opened successfully (fd=%d)\n\n", fd);
    
    // Run tests based on options
    if (run_all || status_only) {
        test_get_status(fd);
    }
    
    if (run_all || events_only) {
        test_get_events(fd);
    }
    
    if (run_all || mode5_level >= 0) {
        if (mode5_level < 0) mode5_level = 1; // Default to Standard for -a
        test_set_mode5(fd, mode5_level);
    }
    
    if (run_all || dsmil_mode >= 0) {
        if (dsmil_mode < 0) dsmil_mode = 1; // Default to Basic for -a
        test_activate_dsmil(fd, dsmil_mode);
    }
    
    if (run_all || force_activate) {
        test_force_activate(fd);
    }
    
    if (run_all || test_wipe) {
        test_emergency_wipe(fd, wipe_confirm);
    }
    
    // If running all tests, get status again to see changes
    if (run_all) {
        printf("\nFinal status after tests:\n");
        test_get_status(fd);
    }
    
    // If no specific tests requested, just show status
    if (!run_all && !status_only && !events_only && mode5_level < 0 && 
        dsmil_mode < 0 && !force_activate && !test_wipe) {
        test_get_status(fd);
    }
    
    close(fd);
    printf("\nTest completed.\n");
    return 0;
}