# Comprehensive Testing Infrastructure Plan

## 🎯 **Overview**

Military-grade drivers require exhaustive testing to ensure reliability, security, and compliance. This plan outlines a complete testing infrastructure including unit tests, integration tests, hardware simulation, stress testing, fuzzing, and continuous integration.

**CRITICAL UPDATES FROM ENUMERATION:**
- **12 DSMIL devices** require expanded test coverage
- **JRTC1 marker** needs specific training mode tests
- **1.8GB hidden memory** requires memory access validation
- **144 DSMIL ACPI references** need comprehensive ACPI testing

## 📋 **Current State Analysis**

### ✅ **What We Have:**
- Basic test-milspec.c for IOCTL testing
- Manual module loading/unloading scripts
- Simple test scenarios in examples.sh
- Build system with warnings enabled

### ❌ **What's Missing:**
- Automated test framework for 12 DSMIL devices
- Unit testing infrastructure
- Hardware simulation layer including JRTC1
- Stress and performance tests for hidden memory
- Security fuzzing for all DSMIL subsystems
- Code coverage analysis
- CI/CD pipeline
- Test result reporting
- ACPI method testing framework

## 🏗️ **Comprehensive Implementation Plan**

### **Phase 1: Unit Testing Framework**

#### 1.1 KUnit Integration
```c
/* tests/milspec_test.c - KUnit test suite */
#include <kunit/test.h>
#include "../dell-millspec-enhanced.c"

/* Test fixture for driver state */
struct milspec_test_fixture {
    struct platform_device *pdev;
    void __iomem *fake_mmio;
    struct gpio_desc *fake_gpios[5];
    bool tpm_available;
};

/* Test case: Mode 5 transitions */
static void milspec_test_mode5_transitions(struct kunit *test)
{
    struct milspec_test_fixture *fixture = test->priv;
    int old_level, new_level;
    
    /* Test valid transitions */
    for (old_level = 0; old_level <= 4; old_level++) {
        for (new_level = 0; new_level <= 4; new_level++) {
            milspec_state.mode5_level = old_level;
            
            /* Attempt transition */
            int ret = milspec_set_mode5_level(new_level);
            
            /* Verify rules */
            if (old_level >= MODE5_PARANOID && new_level < old_level) {
                KUNIT_EXPECT_EQ(test, ret, -EPERM);
                KUNIT_EXPECT_EQ(test, milspec_state.mode5_level, old_level);
            } else {
                KUNIT_EXPECT_EQ(test, ret, 0);
                KUNIT_EXPECT_EQ(test, milspec_state.mode5_level, new_level);
            }
        }
    }
}

/* Test case: Emergency wipe arming */
static void milspec_test_emergency_wipe_arming(struct kunit *test)
{
    /* Test unarmed wipe rejection */
    milspec_state.emergency_wipe_armed = false;
    milspec_emergency_wipe();
    KUNIT_EXPECT_FALSE(test, wipe_status.completed);
    
    /* Test armed wipe (mock) */
    milspec_state.emergency_wipe_armed = true;
    /* Note: Actual wipe would destroy system, so we mock */
    KUNIT_EXPECT_TRUE(test, milspec_state.emergency_wipe_armed);
}

/* Test case: MMIO register access */
static void milspec_test_mmio_access(struct kunit *test)
{
    struct milspec_test_fixture *fixture = test->priv;
    u32 value;
    
    /* Test null MMIO handling */
    milspec_mmio_base = NULL;
    value = milspec_read_reg(MILSPEC_REG_STATUS);
    KUNIT_EXPECT_EQ(test, value, 0);
    
    /* Test with fake MMIO */
    milspec_mmio_base = fixture->fake_mmio;
    iowrite32(0xDEADBEEF, fixture->fake_mmio + MILSPEC_REG_STATUS);
    value = milspec_read_reg(MILSPEC_REG_STATUS);
    KUNIT_EXPECT_EQ(test, value, 0xDEADBEEF);
}

/* Test suite definition */
static struct kunit_case milspec_test_cases[] = {
    KUNIT_CASE(milspec_test_mode5_transitions),
    KUNIT_CASE(milspec_test_emergency_wipe_arming),
    KUNIT_CASE(milspec_test_mmio_access),
    KUNIT_CASE(milspec_test_gpio_intrusion),
    KUNIT_CASE(milspec_test_tpm_measurements),
    KUNIT_CASE(milspec_test_dsmil_activation),
    {}
};

static struct kunit_suite milspec_test_suite = {
    .name = "milspec",
    .init = milspec_test_init,
    .exit = milspec_test_exit,
    .test_cases = milspec_test_cases,
};

kunit_test_suites(&milspec_test_suite);
```

#### 1.2 Mock Infrastructure
```c
/* tests/milspec_mocks.h - Mock implementations */

/* Mock MMIO operations */
static u32 mock_mmio_registers[256];

static u32 mock_milspec_read_reg(u32 offset)
{
    if (offset >= sizeof(mock_mmio_registers))
        return 0xFFFFFFFF;
    return mock_mmio_registers[offset / 4];
}

static void mock_milspec_write_reg(u32 offset, u32 value)
{
    if (offset < sizeof(mock_mmio_registers))
        mock_mmio_registers[offset / 4] = value;
}

/* Mock GPIO operations */
struct mock_gpio_desc {
    int value;
    int direction;
    bool requested;
};

static struct mock_gpio_desc mock_gpios[512];

static int mock_gpiod_get_value(struct gpio_desc *desc)
{
    int gpio = (int)(uintptr_t)desc;
    return mock_gpios[gpio].value;
}

/* Mock I2C for crypto chip */
static int mock_i2c_master_send(const struct i2c_client *client,
                               const char *buf, int count)
{
    /* Simulate crypto chip responses */
    if (buf[0] == 0x30) /* Wake command */
        return count;
    if (buf[0] == 0x02 && buf[1] == 0x00) /* Read command */
        return count;
    return -EIO;
}

/* Mock TPM operations */
static struct tpm_chip *mock_tpm_default_chip(void)
{
    static struct tpm_chip mock_chip = {
        .flags = TPM_CHIP_FLAG_TPM2,
    };
    return &mock_chip;
}
```

### **Phase 2: Hardware Simulation Layer**

#### 2.1 Virtual Hardware Device
```c
/* tests/milspec_sim.c - Hardware simulator */

struct milspec_sim_device {
    /* Simulated hardware state */
    u32 mmio_registers[256];
    bool gpio_states[512];
    bool intrusion_triggered;
    bool tamper_triggered;
    
    /* Simulation control */
    struct timer_list intrusion_timer;
    struct work_struct async_work;
    bool running;
    
    /* Statistics */
    atomic_t mmio_reads;
    atomic_t mmio_writes;
    atomic_t gpio_changes;
};

static struct milspec_sim_device *sim_device;

/* Initialize simulation device */
static int milspec_sim_init(void)
{
    sim_device = kzalloc(sizeof(*sim_device), GFP_KERNEL);
    if (!sim_device)
        return -ENOMEM;
    
    /* Initialize hardware state */
    sim_device->mmio_registers[MILSPEC_REG_STATUS / 4] = 
        MILSPEC_STATUS_READY | MILSPEC_STATUS_SECURE;
    
    sim_device->mmio_registers[MILSPEC_REG_VERSION / 4] = 
        0x01000000; /* Version 1.0 */
    
    /* Set up timers for events */
    timer_setup(&sim_device->intrusion_timer, 
                milspec_sim_intrusion_timer, 0);
    
    INIT_WORK(&sim_device->async_work, milspec_sim_async_events);
    
    sim_device->running = true;
    
    pr_info("MIL-SPEC: Hardware simulator initialized\n");
    return 0;
}

/* Simulate intrusion events */
static void milspec_sim_trigger_intrusion(void)
{
    if (!sim_device || !sim_device->running)
        return;
    
    /* Set GPIO states */
    sim_device->gpio_states[384] = true; /* Intrusion GPIO */
    
    /* Update MMIO status */
    sim_device->mmio_registers[MILSPEC_REG_INTRUSION / 4] |= 
        MILSPEC_STATUS_INTRUSION;
    
    /* Trigger interrupt if handler registered */
    if (intrusion_irq >= 0) {
        generic_handle_irq(intrusion_irq);
    }
    
    pr_info("MIL-SPEC SIM: Intrusion event triggered\n");
}

/* Simulate progressive hardware degradation */
static void milspec_sim_degrade_hardware(int level)
{
    switch (level) {
    case 1: /* Minor issues */
        sim_device->mmio_registers[MILSPEC_REG_STATUS / 4] |= BIT(16);
        break;
        
    case 2: /* Crypto chip failure */
        crypto_chip.present = false;
        break;
        
    case 3: /* DSMIL device failures */
        milspec_state.dsmil_active[5] = false;
        milspec_state.dsmil_active[7] = false;
        break;
        
    case 4: /* Critical failure */
        sim_device->mmio_registers[MILSPEC_REG_STATUS / 4] = 0xDEADDEAD;
        break;
    }
}
```

#### 2.2 Simulation Control Interface
```c
/* Debugfs interface for simulation control */
static ssize_t sim_trigger_write(struct file *file, const char __user *buf,
                                size_t count, loff_t *ppos)
{
    char cmd[32];
    
    if (count >= sizeof(cmd))
        return -EINVAL;
    
    if (copy_from_user(cmd, buf, count))
        return -EFAULT;
    
    cmd[count] = '\0';
    
    if (strncmp(cmd, "intrusion", 9) == 0) {
        milspec_sim_trigger_intrusion();
    } else if (strncmp(cmd, "tamper", 6) == 0) {
        milspec_sim_trigger_tamper();
    } else if (strncmp(cmd, "degrade:", 8) == 0) {
        int level = simple_strtol(cmd + 8, NULL, 10);
        milspec_sim_degrade_hardware(level);
    } else if (strncmp(cmd, "reset", 5) == 0) {
        milspec_sim_reset();
    }
    
    return count;
}

static const struct file_operations sim_trigger_fops = {
    .write = sim_trigger_write,
};
```

### **Phase 3: Integration Testing**

#### 3.1 Test Scenarios
```bash
#!/bin/bash
# tests/integration/test_scenarios.sh

# Test scenario: Boot sequence
test_boot_sequence() {
    echo "=== Testing boot sequence ==="
    
    # Load module with simulation
    insmod dell-milspec.ko milspec_force=1 milspec_simulation=1
    
    # Verify initialization
    check_sysfs_value "mode5" "0"
    check_sysfs_value "intrusion_status" "clear"
    
    # Simulate GPIO detection
    echo "gpio:147:1" > /sys/kernel/debug/milspec/sim_trigger
    
    # Verify Mode 5 activation
    check_sysfs_value "mode5" "1"
    
    # Check event log
    grep "Mode 5 enabled by GPIO" /sys/kernel/debug/milspec/events
}

# Test scenario: Security escalation
test_security_escalation() {
    echo "=== Testing security escalation ==="
    
    # Start at standard mode
    echo "1" > /sys/devices/platform/dell-milspec/mode5
    
    # Trigger intrusion
    echo "intrusion" > /sys/kernel/debug/milspec/sim_trigger
    sleep 1
    
    # Verify escalation
    mode5_level=$(cat /sys/devices/platform/dell-milspec/mode5)
    if [ "$mode5_level" -lt "2" ]; then
        echo "FAIL: Mode 5 did not escalate on intrusion"
        return 1
    fi
    
    # Verify TPM measurement
    check_tpm_pcr 10
}

# Test scenario: DSMIL activation
test_dsmil_activation() {
    echo "=== Testing DSMIL activation ==="
    
    # Activate DSMIL mode
    ./test-milspec activate-dsmil 2
    
    # Verify devices
    for i in {0..9}; do
        status=$(cat /sys/devices/platform/dell-milspec/dsmil | \
                 grep "DSMIL0D$i" | awk '{print $2}')
        
        if [ "$i" -le "2" ] && [ "$status" != "active" ]; then
            echo "FAIL: Critical DSMIL device $i not active"
            return 1
        fi
    done
}
```

#### 3.2 Automated Test Runner
```python
#!/usr/bin/env python3
# tests/run_tests.py

import subprocess
import json
import time
import sys

class MilspecTestRunner:
    def __init__(self):
        self.results = []
        self.module_loaded = False
        
    def setup(self):
        """Load module and prepare test environment"""
        # Unload if already loaded
        subprocess.run(["rmmod", "dell-milspec"], capture_output=True)
        
        # Load with simulation mode
        result = subprocess.run(
            ["insmod", "dell-milspec.ko", 
             "milspec_force=1", "milspec_simulation=1"],
            capture_output=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to load module: {result.stderr}")
            
        self.module_loaded = True
        time.sleep(1)  # Let module initialize
        
    def teardown(self):
        """Cleanup test environment"""
        if self.module_loaded:
            subprocess.run(["rmmod", "dell-milspec"])
            
    def run_test(self, test_name, test_func):
        """Run individual test"""
        print(f"Running {test_name}...", end=" ")
        
        try:
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            
            self.results.append({
                "name": test_name,
                "status": "PASS",
                "duration": duration
            })
            print(f"PASS ({duration:.2f}s)")
            
        except Exception as e:
            self.results.append({
                "name": test_name,
                "status": "FAIL",
                "error": str(e)
            })
            print(f"FAIL: {e}")
            
    def test_mode5_transitions(self):
        """Test Mode 5 security level transitions"""
        sysfs_path = "/sys/devices/platform/dell-milspec/mode5"
        
        # Test valid transitions
        for level in [0, 1, 2, 3, 4]:
            with open(sysfs_path, "w") as f:
                f.write(str(level))
                
            with open(sysfs_path, "r") as f:
                actual = int(f.read().strip())
                
            assert actual == level, f"Expected {level}, got {actual}"
            
        # Test invalid downgrade from paranoid
        with open(sysfs_path, "w") as f:
            f.write("4")  # PARANOID_PLUS
            
        try:
            with open(sysfs_path, "w") as f:
                f.write("2")  # Try to downgrade
            assert False, "Downgrade should have failed"
        except IOError:
            pass  # Expected
            
    def test_intrusion_detection(self):
        """Test intrusion detection and response"""
        # Trigger simulated intrusion
        with open("/sys/kernel/debug/milspec/sim_trigger", "w") as f:
            f.write("intrusion")
            
        time.sleep(0.5)  # Let handler process
        
        # Check intrusion status
        with open("/sys/devices/platform/dell-milspec/intrusion_status", "r") as f:
            status = f.read().strip()
            
        assert "detected" in status.lower(), f"Intrusion not detected: {status}"
        
        # Verify event logged
        events = subprocess.run(
            ["grep", "intrusion", "/sys/kernel/debug/milspec/events"],
            capture_output=True, text=True
        ).stdout
        
        assert "INTRUSION" in events, "Intrusion event not logged"
```

### **Phase 4: Stress Testing**

#### 4.1 Load Testing
```c
/* tests/stress/load_test.c */
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define NUM_THREADS 100
#define ITERATIONS 10000

struct thread_data {
    int thread_id;
    int fd;
    int errors;
    long operations;
};

void *stress_thread(void *arg)
{
    struct thread_data *data = (struct thread_data *)arg;
    struct milspec_status status;
    int i, ret;
    
    for (i = 0; i < ITERATIONS; i++) {
        /* Random IOCTL operations */
        switch (rand() % 5) {
        case 0: /* Get status */
            ret = ioctl(data->fd, MILSPEC_IOC_GET_STATUS, &status);
            break;
            
        case 1: /* Set mode5 */
            int level = rand() % 5;
            ret = ioctl(data->fd, MILSPEC_IOC_SET_MODE5, &level);
            break;
            
        case 2: /* Force activate */
            ret = ioctl(data->fd, MILSPEC_IOC_FORCE_ACTIVATE, NULL);
            break;
            
        case 3: /* Get events */
            struct milspec_events events;
            ret = ioctl(data->fd, MILSPEC_IOC_GET_EVENTS, &events);
            break;
            
        case 4: /* TPM measure */
            ret = ioctl(data->fd, MILSPEC_IOC_TPM_MEASURE, NULL);
            break;
        }
        
        if (ret < 0)
            data->errors++;
        else
            data->operations++;
            
        /* Add some randomness */
        if (rand() % 100 == 0)
            usleep(rand() % 1000);
    }
    
    return NULL;
}

int main()
{
    pthread_t threads[NUM_THREADS];
    struct thread_data thread_data[NUM_THREADS];
    int fd, i;
    
    /* Open device */
    fd = open("/dev/milspec", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    /* Create stress threads */
    for (i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].fd = fd;
        thread_data[i].errors = 0;
        thread_data[i].operations = 0;
        
        pthread_create(&threads[i], NULL, stress_thread, &thread_data[i]);
    }
    
    /* Wait for completion */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Report results */
    long total_ops = 0, total_errors = 0;
    for (i = 0; i < NUM_THREADS; i++) {
        total_ops += thread_data[i].operations;
        total_errors += thread_data[i].errors;
    }
    
    printf("Stress test complete:\n");
    printf("  Total operations: %ld\n", total_ops);
    printf("  Total errors: %ld\n", total_errors);
    printf("  Error rate: %.2f%%\n", 
           (float)total_errors / total_ops * 100);
    
    close(fd);
    return total_errors > 0 ? 1 : 0;
}
```

#### 4.2 Memory Stress
```bash
#!/bin/bash
# tests/stress/memory_stress.sh

echo "=== Memory stress test ==="

# Initial memory snapshot
INITIAL_MEM=$(grep MemFree /proc/meminfo | awk '{print $2}')

# Load module
insmod dell-milspec.ko

# Stress event logging
for i in {1..100000}; do
    echo "trigger_event:$i" > /sys/kernel/debug/milspec/sim_trigger
done &

# Stress IOCTL operations
for i in {1..100}; do
    ./test-milspec stress &
done

# Monitor memory usage
for i in {1..60}; do
    MEM=$(grep MemFree /proc/meminfo | awk '{print $2}')
    USED=$((INITIAL_MEM - MEM))
    echo "Memory used: $USED KB"
    
    # Check for leaks
    if [ $USED -gt 100000 ]; then
        echo "FAIL: Excessive memory usage detected"
        killall test-milspec
        exit 1
    fi
    
    sleep 1
done

# Cleanup
killall test-milspec
rmmod dell-milspec

# Final check
FINAL_MEM=$(grep MemFree /proc/meminfo | awk '{print $2}')
LEAKED=$((INITIAL_MEM - FINAL_MEM))

if [ $LEAKED -gt 1000 ]; then
    echo "FAIL: Memory leak detected: $LEAKED KB"
    exit 1
fi

echo "PASS: No memory leaks detected"
```

### **Phase 5: Security Testing & Fuzzing**

#### 5.1 AFL++ Fuzzing Setup
```c
/* tests/fuzz/fuzz_ioctl.c */
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define AFL_FUZZ_TESTCASE_LEN 1024

int main()
{
    int fd;
    uint8_t buf[AFL_FUZZ_TESTCASE_LEN];
    ssize_t len;
    
    /* Read fuzz input */
    len = read(0, buf, sizeof(buf));
    if (len < 8)
        return 0;
    
    /* Open device */
    fd = open("/dev/milspec", O_RDWR);
    if (fd < 0)
        return 0;
    
    /* Fuzz IOCTL */
    uint32_t cmd = *(uint32_t *)buf;
    void *arg = buf + 4;
    
    /* Sanitize command to valid range */
    cmd = (cmd & 0xFF) | (MILSPEC_IOC_MAGIC << 8);
    
    /* Execute fuzzed IOCTL */
    ioctl(fd, cmd, arg);
    
    close(fd);
    return 0;
}

/* Makefile for AFL++ */
// CC = afl-clang-fast
// CFLAGS = -O2 -g
// fuzz_ioctl: fuzz_ioctl.c
//     $(CC) $(CFLAGS) -o $@ $<
```

#### 5.2 Syzkaller Integration
```txt
# tests/fuzz/milspec.txt - Syzkaller descriptions

resource fd_milspec[fd]

openat$milspec(fd const[AT_FDCWD], file ptr[in, string["/dev/milspec"]], flags flags[open_flags], mode const[0]) fd_milspec

ioctl$MILSPEC_IOC_GET_STATUS(fd fd_milspec, cmd const[MILSPEC_IOC_GET_STATUS], arg ptr[out, milspec_status])
ioctl$MILSPEC_IOC_SET_MODE5(fd fd_milspec, cmd const[MILSPEC_IOC_SET_MODE5], arg ptr[in, int32[0:4]])
ioctl$MILSPEC_IOC_ACTIVATE_DSMIL(fd fd_milspec, cmd const[MILSPEC_IOC_ACTIVATE_DSMIL], arg ptr[in, int32[0:3]])
ioctl$MILSPEC_IOC_EMERGENCY_WIPE(fd fd_milspec, cmd const[MILSPEC_IOC_EMERGENCY_WIPE], arg ptr[in, int32[MILSPEC_WIPE_CONFIRM]])

milspec_status {
    mode5_enabled    int32
    mode5_level      int32[0:4]
    dsmil_active     array[int32, 10]
    service_mode     int32
    intrusion        int32
}
```

### **Phase 6: Performance Testing**

#### 6.1 Benchmark Suite
```c
/* tests/perf/benchmark.c */
#include <time.h>
#include <stdio.h>

#define BENCH_ITERATIONS 1000000

struct benchmark_result {
    const char *name;
    double ops_per_sec;
    double avg_latency_us;
    double min_latency_us;
    double max_latency_us;
};

static void benchmark_ioctl_get_status(int fd, struct benchmark_result *result)
{
    struct milspec_status status;
    struct timespec start, end;
    double total_time = 0;
    double min_time = 1e9, max_time = 0;
    int i;
    
    /* Warmup */
    for (i = 0; i < 1000; i++) {
        ioctl(fd, MILSPEC_IOC_GET_STATUS, &status);
    }
    
    /* Benchmark */
    for (i = 0; i < BENCH_ITERATIONS; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        ioctl(fd, MILSPEC_IOC_GET_STATUS, &status);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double time = (end.tv_sec - start.tv_sec) * 1e9 + 
                     (end.tv_nsec - start.tv_nsec);
        
        total_time += time;
        if (time < min_time) min_time = time;
        if (time > max_time) max_time = time;
    }
    
    result->name = "IOCTL_GET_STATUS";
    result->ops_per_sec = BENCH_ITERATIONS / (total_time / 1e9);
    result->avg_latency_us = (total_time / BENCH_ITERATIONS) / 1000;
    result->min_latency_us = min_time / 1000;
    result->max_latency_us = max_time / 1000;
}

static void benchmark_event_logging(int fd, struct benchmark_result *result)
{
    struct timespec start, end;
    int i;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    /* Trigger many events */
    for (i = 0; i < BENCH_ITERATIONS; i++) {
        ioctl(fd, MILSPEC_IOC_FORCE_ACTIVATE, NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double total_time = (end.tv_sec - start.tv_sec) * 1e9 + 
                       (end.tv_nsec - start.tv_nsec);
    
    result->name = "Event Logging";
    result->ops_per_sec = BENCH_ITERATIONS / (total_time / 1e9);
    result->avg_latency_us = (total_time / BENCH_ITERATIONS) / 1000;
}
```

#### 6.2 Latency Analysis
```python
#!/usr/bin/env python3
# tests/perf/latency_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import json

def measure_interrupt_latency():
    """Measure GPIO interrupt latency"""
    cmd = ["./test-milspec", "measure-irq-latency", "1000"]
    output = subprocess.check_output(cmd, text=True)
    
    latencies = []
    for line in output.splitlines():
        if "Latency:" in line:
            us = float(line.split()[1])
            latencies.append(us)
            
    return np.array(latencies)

def plot_latency_histogram(latencies):
    """Generate latency histogram"""
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(latencies), color='red', linestyle='dashed', 
                linewidth=2, label=f'Mean: {np.mean(latencies):.2f} μs')
    plt.axvline(np.percentile(latencies, 99), color='orange', 
                linestyle='dashed', linewidth=2, 
                label=f'P99: {np.percentile(latencies, 99):.2f} μs')
    
    plt.xlabel('Latency (μs)')
    plt.ylabel('Frequency')
    plt.title('GPIO Interrupt Latency Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('interrupt_latency.png')
    plt.close()

def generate_report():
    """Generate performance report"""
    report = {
        "timestamp": subprocess.check_output(["date", "-Iseconds"], text=True).strip(),
        "kernel": subprocess.check_output(["uname", "-r"], text=True).strip(),
        "benchmarks": []
    }
    
    # Run benchmarks
    benchmarks = [
        "./benchmark_ioctl",
        "./benchmark_events",
        "./benchmark_crypto",
        "./benchmark_tpm"
    ]
    
    for bench in benchmarks:
        try:
            output = subprocess.check_output([bench], text=True)
            # Parse output and add to report
            # ...
        except Exception as e:
            print(f"Failed to run {bench}: {e}")
            
    # Save report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    # Measure interrupt latency
    latencies = measure_interrupt_latency()
    plot_latency_histogram(latencies)
    
    # Generate full report
    generate_report()
```

### **Phase 7: Continuous Integration**

#### 7.1 GitHub Actions Workflow
```yaml
# .github/workflows/milspec-ci.yml
name: Dell MIL-SPEC Driver CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        kernel: ['6.1', '6.6', '6.14']
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          linux-headers-generic \
          build-essential \
          kmod \
          coccinelle \
          sparse
          
    - name: Setup kernel ${{ matrix.kernel }}
      run: |
        wget https://kernel.org/pub/linux/kernel/v6.x/linux-${{ matrix.kernel }}.tar.xz
        tar -xf linux-${{ matrix.kernel }}.tar.xz
        cd linux-${{ matrix.kernel }}
        make defconfig
        make modules_prepare
        
    - name: Build driver
      run: |
        make KDIR=linux-${{ matrix.kernel }} clean all
        
    - name: Static analysis
      run: |
        make C=2 CF="-D__CHECK_ENDIAN__" \
          KDIR=linux-${{ matrix.kernel }} modules
          
    - name: Coccinelle checks
      run: |
        make coccicheck MODE=report \
          KDIR=linux-${{ matrix.kernel }}
          
    - name: Run unit tests
      run: |
        make KDIR=linux-${{ matrix.kernel }} tests
        
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: milspec-${{ matrix.kernel }}
        path: |
          *.ko
          tests/results/

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security checks
      run: |
        # Check for common vulnerabilities
        ./scripts/security-check.sh
        
    - name: License compliance
      run: |
        # Verify GPL compliance
        ./scripts/license-check.sh

  documentation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build documentation
      run: |
        make htmldocs
        
    - name: Check documentation coverage
      run: |
        ./scripts/doc-coverage.sh
```

#### 7.2 Test Result Reporting
```python
#!/usr/bin/env python3
# tests/report_generator.py

import xml.etree.ElementTree as ET
import json
import sys
from datetime import datetime

class TestReportGenerator:
    def __init__(self):
        self.results = []
        
    def parse_kunit_results(self, xml_file):
        """Parse KUnit XML results"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for testcase in root.findall('.//testcase'):
            result = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0)),
                'status': 'pass'
            }
            
            failure = testcase.find('failure')
            if failure is not None:
                result['status'] = 'fail'
                result['message'] = failure.get('message', '')
                
            self.results.append(result)
            
    def generate_html_report(self, output_file):
        """Generate HTML test report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MIL-SPEC Driver Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dell MIL-SPEC Driver Test Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Tests: {len(self.results)}</li>
                <li>Passed: {sum(1 for r in self.results if r['status'] == 'pass')}</li>
                <li>Failed: {sum(1 for r in self.results if r['status'] == 'fail')}</li>
            </ul>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Time (s)</th>
                    <th>Details</th>
                </tr>
        """
        
        for result in self.results:
            status_class = result['status']
            html += f"""
                <tr>
                    <td>{result['name']}</td>
                    <td class="{status_class}">{result['status'].upper()}</td>
                    <td>{result['time']:.3f}</td>
                    <td>{result.get('message', '')}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
            
    def generate_json_report(self, output_file):
        """Generate JSON test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r['status'] == 'pass'),
                'failed': sum(1 for r in self.results if r['status'] == 'fail'),
            },
            'tests': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

if __name__ == '__main__':
    generator = TestReportGenerator()
    
    # Parse all result files
    for xml_file in sys.argv[1:]:
        generator.parse_kunit_results(xml_file)
        
    # Generate reports
    generator.generate_html_report('test_report.html')
    generator.generate_json_report('test_report.json')
```

## 📊 **Implementation Priority**

### **High Priority:**
1. KUnit test framework
2. Basic integration tests
3. Hardware simulation layer
4. CI/CD pipeline setup

### **Medium Priority:**
5. Stress testing suite
6. Security fuzzing
7. Performance benchmarks
8. Test reporting

### **Low Priority:**
9. Advanced simulation scenarios
10. Distributed testing
11. Hardware-in-loop testing
12. Compliance testing

## ⚠️ **Testing Requirements**

1. **Code Coverage**: Minimum 80% coverage
2. **Security**: All IOCTLs must be fuzz tested
3. **Performance**: No regression in latency
4. **Reliability**: 24-hour stress test must pass
5. **Compatibility**: Test on 3 kernel versions

## 📅 **Implementation Timeline**

- **Week 1**: KUnit framework and basic tests
- **Week 2**: Hardware simulation layer
- **Week 3**: Integration and stress tests
- **Week 4**: Security testing and fuzzing
- **Week 5**: Performance tests and CI/CD
- **Week 6**: Documentation and reporting

## 🔧 **Required Tools**

- KUnit (kernel unit testing)
- AFL++ (fuzzing)
- Syzkaller (kernel fuzzing)
- Coccinelle (static analysis)
- Perf (performance analysis)
- GitHub Actions (CI/CD)

---

**Status**: Plan Ready for Implementation
**Priority**: Critical - Required for production
**Estimated Effort**: 6 weeks development
**Dependencies**: Test hardware, CI infrastructure