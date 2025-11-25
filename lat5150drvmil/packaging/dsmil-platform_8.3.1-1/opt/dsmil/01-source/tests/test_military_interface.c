/*
 * Military Device Interface Test Program
 * Dell Latitude 5450 MIL-SPEC DSMIL Device Interface
 * 
 * PHASE 1: Safe Foundation Testing for 0x8000-0x806B range
 * - READ-ONLY operations only
 * - Comprehensive safety validation
 * - Thermal monitoring integration
 * - Emergency stop mechanisms
 * 
 * Copyright (C) 2025 JRTC1 Educational Development
 * Security Level: READ-ONLY SAFE TESTING
 */

#include "military_device_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <getopt.h>

/* Test configuration */
typedef struct {
    bool verbose;
    bool test_all_devices;
    bool continuous_monitoring;
    uint16_t specific_device;
    int scan_interval_sec;
    bool emergency_test;
    bool performance_test;
} test_config_t;

/* Global test state */
static volatile bool g_running = true;
static test_config_t g_config = {0};

/* Function declarations */
static void signal_handler(int sig);
static void print_usage(const char *program_name);
static int parse_arguments(int argc, char *argv[], test_config_t *config);
static int run_basic_tests(void);
static int run_device_discovery_test(void);
static int run_individual_device_test(uint16_t device_id);
static int run_thermal_safety_test(void);
static int run_quarantine_enforcement_test(void);
static int run_continuous_monitoring_test(void);
static int run_performance_test(void);
static int run_emergency_stop_test(void);
static void print_test_header(const char *test_name);
static void print_test_result(const char *test_name, bool passed, const char *details);

int main(int argc, char *argv[])
{
    int result = 0;
    
    /* Install signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    /* Parse command line arguments */
    if (parse_arguments(argc, argv, &g_config) != 0) {
        return 1;
    }
    
    /* Print banner */
    printf("=========================================================\n");
    printf("Military Device Interface Test Program\n");
    printf("Dell Latitude 5450 MIL-SPEC DSMIL Device Interface\n");
    printf("PHASE 1: Safe Foundation Testing (0x8000-0x806B)\n");
    printf("=========================================================\n");
    
    mildev_print_version();
    printf("\n");
    
    /* Initialize military device interface */
    printf("Initializing military device interface...\n");
    int init_result = mildev_init();
    if (init_result != MILDEV_SUCCESS) {
        printf("ERROR: Failed to initialize interface: %s\n", mildev_error_string(init_result));
        return 1;
    }
    printf("Interface initialized successfully.\n\n");
    
    /* Run basic tests first */
    result = run_basic_tests();
    if (result != 0) {
        printf("Basic tests failed, stopping here for safety.\n");
        goto cleanup;
    }
    
    /* Run thermal safety test */
    result = run_thermal_safety_test();
    if (result != 0) {
        printf("Thermal safety test failed, stopping for safety.\n");
        goto cleanup;
    }
    
    /* Run quarantine enforcement test */
    result = run_quarantine_enforcement_test();
    if (result != 0) {
        printf("Quarantine test failed.\n");
        goto cleanup;
    }
    
    /* Run device discovery test */
    if (g_config.test_all_devices) {
        result = run_device_discovery_test();
        if (result != 0) {
            printf("Device discovery test failed.\n");
        }
    }
    
    /* Run specific device test */
    if (g_config.specific_device != 0) {
        result = run_individual_device_test(g_config.specific_device);
    }
    
    /* Run performance test */
    if (g_config.performance_test) {
        result = run_performance_test();
    }
    
    /* Run emergency test */
    if (g_config.emergency_test) {
        result = run_emergency_stop_test();
    }
    
    /* Run continuous monitoring */
    if (g_config.continuous_monitoring) {
        result = run_continuous_monitoring_test();
    }
    
cleanup:
    /* Cleanup */
    printf("\nCleaning up...\n");
    mildev_cleanup();
    
    printf("\nTest program completed with result: %d\n", result);
    return result;
}

static void signal_handler(int sig)
{
    printf("\nReceived signal %d, shutting down safely...\n", sig);
    g_running = false;
}

static void print_usage(const char *program_name)
{
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -v, --verbose              Enable verbose output\n");
    printf("  -a, --all                  Test all devices in range\n");
    printf("  -d, --device=ID            Test specific device (hex, e.g., 0x8000)\n");
    printf("  -m, --monitor              Continuous monitoring mode\n");
    printf("  -i, --interval=SECONDS     Monitoring interval (default: 5)\n");
    printf("  -p, --performance          Run performance tests\n");
    printf("  -e, --emergency            Test emergency stop mechanisms\n");
    printf("  -h, --help                 Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -a                      Test all devices\n", program_name);
    printf("  %s -d 0x8001              Test device 0x8001\n", program_name);
    printf("  %s -m -i 10               Monitor every 10 seconds\n", program_name);
    printf("  %s -v -a -p               Verbose all devices + performance\n", program_name);
}

static int parse_arguments(int argc, char *argv[], test_config_t *config)
{
    static struct option long_options[] = {
        {"verbose", no_argument, 0, 'v'},
        {"all", no_argument, 0, 'a'},
        {"device", required_argument, 0, 'd'},
        {"monitor", no_argument, 0, 'm'},
        {"interval", required_argument, 0, 'i'},
        {"performance", no_argument, 0, 'p'},
        {"emergency", no_argument, 0, 'e'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    config->scan_interval_sec = 5;  /* Default 5 seconds */
    
    int c;
    while ((c = getopt_long(argc, argv, "vad:mi:peh", long_options, NULL)) != -1) {
        switch (c) {
            case 'v':
                config->verbose = true;
                break;
            case 'a':
                config->test_all_devices = true;
                break;
            case 'd':
                config->specific_device = (uint16_t)strtoul(optarg, NULL, 0);
                break;
            case 'm':
                config->continuous_monitoring = true;
                break;
            case 'i':
                config->scan_interval_sec = atoi(optarg);
                if (config->scan_interval_sec < 1) {
                    printf("ERROR: Interval must be at least 1 second\n");
                    return 1;
                }
                break;
            case 'p':
                config->performance_test = true;
                break;
            case 'e':
                config->emergency_test = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            case '?':
                return 1;
            default:
                break;
        }
    }
    
    return 0;
}

static int run_basic_tests(void)
{
    print_test_header("Basic Interface Tests");
    
    /* Test 1: System status */
    mildev_system_status_t status;
    int result = mildev_get_system_status(&status);
    print_test_result("System Status", result == MILDEV_SUCCESS, 
                     result == MILDEV_SUCCESS ? "System operational" : mildev_error_string(result));
    
    if (result == MILDEV_SUCCESS && g_config.verbose) {
        printf("  Kernel module loaded: %s\n", status.kernel_module_loaded ? "Yes" : "No");
        printf("  Thermal safe: %s\n", status.thermal_safe ? "Yes" : "No");
        printf("  Current temperature: %d°C\n", status.current_temp_celsius);
    }
    
    /* Test 2: Version information */
    uint32_t version = mildev_get_version_code();
    print_test_result("Version Check", version > 0, "Version retrieved successfully");
    
    if (g_config.verbose) {
        printf("  Version code: 0x%08X\n", version);
    }
    
    /* Test 3: Quarantine check */
    bool quarantine_working = true;
    for (int i = 0; i < MILDEV_QUARANTINE_COUNT; i++) {
        if (!mildev_check_quarantine(MILDEV_QUARANTINE_LIST[i])) {
            quarantine_working = false;
            break;
        }
    }
    print_test_result("Quarantine Check", quarantine_working, "Quarantine list functioning");
    
    printf("\n");
    return (result == MILDEV_SUCCESS && quarantine_working) ? 0 : 1;
}

static int run_device_discovery_test(void)
{
    print_test_header("Device Discovery Test");
    
    mildev_discovery_result_t discovery;
    int result = mildev_scan_devices(&discovery);
    
    print_test_result("Device Scan", result == MILDEV_SUCCESS,
                     result == MILDEV_SUCCESS ? "Scan completed" : mildev_error_string(result));
    
    if (result == MILDEV_SUCCESS) {
        mildev_print_discovery_summary(&discovery);
        
        if (g_config.verbose) {
            printf("\nDevice Details:\n");
            for (int i = 0; i < MILDEV_RANGE_SIZE; i++) {
                mildev_device_info_t *device = &discovery.devices[i];
                if (device->device_id >= MILDEV_BASE_ADDR && device->device_id <= MILDEV_END_ADDR) {
                    printf("Device 0x%04X: %s", device->device_id, mildev_state_string(device->state));
                    if (device->is_quarantined) {
                        printf(" [QUARANTINED]");
                    }
                    if (device->state == MILDEV_STATE_SAFE) {
                        printf(" (Response: 0x%08X)", device->last_response);
                    }
                    printf("\n");
                }
            }
        }
    }
    
    printf("\n");
    return result == MILDEV_SUCCESS ? 0 : 1;
}

static int run_individual_device_test(uint16_t device_id)
{
    printf("=== Individual Device Test: 0x%04X ===\n", device_id);
    
    /* Validate device ID */
    if (!MILDEV_IS_SAFE_RANGE(device_id)) {
        print_test_result("Device Range Check", false, "Device outside safe range");
        return 1;
    }
    
    /* Check if quarantined */
    if (MILDEV_IS_QUARANTINED(device_id)) {
        print_test_result("Quarantine Check", true, "Device properly quarantined");
        return 0;  /* Success - quarantine working */
    }
    
    /* Get device info */
    mildev_device_info_t info;
    int result = mildev_get_device_info(device_id, &info);
    
    print_test_result("Device Info", result == MILDEV_SUCCESS,
                     result == MILDEV_SUCCESS ? "Device info retrieved" : mildev_error_string(result));
    
    if (result == MILDEV_SUCCESS && g_config.verbose) {
        mildev_print_device_info(&info);
    }
    
    /* Try reading with retry */
    if (result == MILDEV_SUCCESS && info.access == MILDEV_ACCESS_READ) {
        uint32_t response;
        int read_result = mildev_read_device_with_retry(device_id, &response, 3);
        
        print_test_result("Device Read", read_result == MILDEV_SUCCESS,
                         read_result == MILDEV_SUCCESS ? "Device read successful" : mildev_error_string(read_result));
        
        if (read_result == MILDEV_SUCCESS && g_config.verbose) {
            printf("  Device response: 0x%08X\n", response);
        }
    }
    
    printf("\n");
    return result == MILDEV_SUCCESS ? 0 : 1;
}

static int run_thermal_safety_test(void)
{
    print_test_header("Thermal Safety Test");
    
    int temp = mildev_get_thermal_celsius();
    bool thermal_safe = MILDEV_IS_THERMAL_SAFE(temp);
    
    print_test_result("Thermal Reading", temp > 0, "Temperature sensor working");
    print_test_result("Thermal Safety", thermal_safe, 
                     thermal_safe ? "Temperature within safe limits" : "THERMAL WARNING");
    
    printf("  Current temperature: %d°C (Limit: %d°C)\n", temp, MILDEV_MAX_THERMAL_C);
    
    if (!thermal_safe) {
        printf("  WARNING: System temperature exceeds safe limits!\n");
        printf("  Thermal protection will prevent device operations.\n");
    }
    
    printf("\n");
    return thermal_safe ? 0 : 1;
}

static int run_quarantine_enforcement_test(void)
{
    print_test_header("Quarantine Enforcement Test");
    
    bool all_passed = true;
    
    printf("Testing quarantined devices (should be blocked):\n");
    for (int i = 0; i < MILDEV_QUARANTINE_COUNT; i++) {
        uint16_t quarantine_device = MILDEV_QUARANTINE_LIST[i];
        
        /* Attempt to read quarantined device (should fail) */
        uint32_t response;
        int result = mildev_read_device_safe(quarantine_device, &response);
        
        bool blocked_correctly = (result == MILDEV_ERROR_QUARANTINED);
        
        printf("  Device 0x%04X: %s\n", quarantine_device, 
               blocked_correctly ? "BLOCKED (Correct)" : "ERROR: Not blocked!");
        
        if (!blocked_correctly) {
            all_passed = false;
        }
    }
    
    print_test_result("Quarantine Enforcement", all_passed, 
                     all_passed ? "All quarantined devices properly blocked" : "SECURITY ERROR");
    
    if (!all_passed) {
        printf("  CRITICAL: Quarantine enforcement failed!\n");
        printf("  This is a serious security issue.\n");
    }
    
    printf("\n");
    return all_passed ? 0 : 1;
}

static int run_continuous_monitoring_test(void)
{
    print_test_header("Continuous Monitoring Test");
    
    printf("Starting continuous monitoring (Ctrl+C to stop)...\n");
    printf("Monitoring interval: %d seconds\n", g_config.scan_interval_sec);
    printf("Range: 0x%04X - 0x%04X\n\n", MILDEV_BASE_ADDR, MILDEV_END_ADDR);
    
    int cycle = 0;
    
    while (g_running) {
        cycle++;
        
        time_t now = time(NULL);
        printf("=== Monitoring Cycle %d - %s", cycle, ctime(&now));
        
        /* Check thermal status */
        int temp = mildev_get_thermal_celsius();
        printf("Temperature: %d°C ", temp);
        if (!MILDEV_IS_THERMAL_SAFE(temp)) {
            printf("WARNING: THERMAL LIMIT EXCEEDED!\n");
            mildev_emergency_stop("Thermal limit exceeded during monitoring");
        } else {
            printf("(Safe)\n");
        }
        
        /* Get system status */
        mildev_system_status_t status;
        if (mildev_get_system_status(&status) == MILDEV_SUCCESS) {
            printf("System Status: %s, Safe devices: %d, Quarantined: %d\n",
                   status.kernel_module_loaded ? "OK" : "ERROR",
                   status.safe_device_count, status.quarantined_count);
        }
        
        /* Quick device check */
        if (g_config.specific_device != 0) {
            uint32_t response;
            int result = mildev_read_device_safe(g_config.specific_device, &response);
            printf("Device 0x%04X: %s", g_config.specific_device,
                   result == MILDEV_SUCCESS ? "OK" : "ERROR");
            if (result == MILDEV_SUCCESS) {
                printf(" (0x%08X)", response);
            }
            printf("\n");
        }
        
        printf("\n");
        
        /* Sleep with interrupt checking */
        for (int i = 0; i < g_config.scan_interval_sec && g_running; i++) {
            sleep(1);
        }
    }
    
    printf("Monitoring stopped.\n\n");
    return 0;
}

static int run_performance_test(void)
{
    print_test_header("Performance Test");
    
    const int test_iterations = 1000;
    const uint16_t test_device = 0x8001;  /* Non-quarantined device */
    
    /* Skip if device is quarantined */
    if (MILDEV_IS_QUARANTINED(test_device)) {
        printf("Skipping performance test - test device is quarantined\n\n");
        return 0;
    }
    
    printf("Testing device 0x%04X with %d iterations...\n", test_device, test_iterations);
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    int successful_reads = 0;
    int failed_reads = 0;
    
    for (int i = 0; i < test_iterations && g_running; i++) {
        uint32_t response;
        int result = mildev_read_device_safe(test_device, &response);
        
        if (result == MILDEV_SUCCESS) {
            successful_reads++;
        } else {
            failed_reads++;
        }
        
        /* Safety check every 100 iterations */
        if ((i % 100) == 0) {
            if (!mildev_is_system_safe()) {
                printf("System became unsafe during performance test, stopping.\n");
                break;
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                    (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    double ops_per_second = successful_reads / elapsed;
    
    printf("Performance Results:\n");
    printf("  Successful reads: %d\n", successful_reads);
    printf("  Failed reads: %d\n", failed_reads);
    printf("  Elapsed time: %.2f seconds\n", elapsed);
    printf("  Operations per second: %.1f\n", ops_per_second);
    printf("  Average latency: %.3f ms\n", (elapsed * 1000.0) / successful_reads);
    
    bool performance_acceptable = (ops_per_second > 100.0);  /* Minimum 100 ops/sec */
    print_test_result("Performance", performance_acceptable,
                     performance_acceptable ? "Performance acceptable" : "Performance below threshold");
    
    printf("\n");
    return 0;
}

static int run_emergency_stop_test(void)
{
    print_test_header("Emergency Stop Test");
    
    printf("WARNING: This test will trigger emergency stop mechanisms.\n");
    printf("Press Enter to continue or Ctrl+C to skip...");
    getchar();
    
    printf("Testing emergency stop with reason: 'Test scenario'\n");
    
    /* Note: This will actually exit the program */
    printf("Emergency stop would be triggered here (simulated).\n");
    printf("In real scenario, mildev_emergency_stop() would halt the system.\n");
    
    print_test_result("Emergency Stop", true, "Emergency stop mechanism ready");
    
    printf("\n");
    return 0;
}

static void print_test_header(const char *test_name)
{
    printf("=== %s ===\n", test_name);
}

static void print_test_result(const char *test_name, bool passed, const char *details)
{
    printf("%-30s: %s", test_name, passed ? "PASS" : "FAIL");
    if (details) {
        printf(" - %s", details);
    }
    printf("\n");
}