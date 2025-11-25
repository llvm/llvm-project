# Comprehensive Hardware Watchdog Support Plan

## 🎯 **Overview**

The hardware watchdog is a critical component for military systems, providing automatic recovery from system hangs, detecting compromised states, and triggering emergency procedures. This plan outlines the implementation of a robust watchdog subsystem integrated with Dell MIL-SPEC hardware.

**CRITICAL UPDATES FROM ENUMERATION:**
- **12 DSMIL devices** require monitoring (not 10)
- **JRTC1 marker** (Junior Reserve Officers' Training Corps) needs special handling
- **1.8GB hidden memory** may contain watchdog state
- **144 DSMIL ACPI references** indicate deep watchdog integration

## 📋 **Current State Analysis**

### ✅ **What We Have:**
- Basic driver framework with timer support
- GPIO intrusion detection with interrupts
- Emergency wipe capability
- Mode 5 security levels
- TPM attestation infrastructure

### ❌ **What's Missing:**
- Hardware watchdog timer integration
- Heartbeat mechanism for 12 DSMIL devices
- Watchdog-triggered emergency actions
- Integration with JRTC1 activation
- Recovery and failsafe procedures
- Hidden memory watchdog state access

## 🏗️ **Comprehensive Implementation Plan**

### **Phase 1: Hardware Watchdog Discovery & Initialization**

#### 1.1 Watchdog Hardware Detection
```c
/* Watchdog timer registers */
#define MILSPEC_REG_WDT_CONTROL    0x80  /* Watchdog control */
#define MILSPEC_REG_WDT_TIMEOUT    0x84  /* Timeout value */
#define MILSPEC_REG_WDT_COUNTER    0x88  /* Current counter */
#define MILSPEC_REG_WDT_STATUS     0x8C  /* Status/interrupt */

/* Control register bits */
#define WDT_CTRL_ENABLE       BIT(0)   /* Enable watchdog */
#define WDT_CTRL_RESET        BIT(1)   /* Reset on timeout */
#define WDT_CTRL_INTERRUPT    BIT(2)   /* Interrupt on timeout */
#define WDT_CTRL_SECURE       BIT(3)   /* Secure mode (no disable) */
#define WDT_CTRL_CASCADE      BIT(4)   /* Cascade with TPM watchdog */

/* Timeout actions */
#define WDT_ACTION_WARN       0x01      /* Warning only */
#define WDT_ACTION_RESET      0x02      /* System reset */
#define WDT_ACTION_POWEROFF   0x03      /* Power off */
#define WDT_ACTION_WIPE       0x04      /* Emergency wipe */
```

#### 1.2 ACPI Watchdog Methods
```c
/* ACPI methods for watchdog control */
static int milspec_wdt_acpi_init(void)
{
    /* _SB.WDOG._STA - Status */
    /* _SB.WDOG._INI - Initialize */
    /* _SB.WDOG.FEED - Feed watchdog */
    /* _SB.WDOG.SETO - Set timeout */
    /* _SB.WDOG.ACTN - Set action */
}
```

### **Phase 2: Linux Watchdog Framework Integration**

#### 2.1 Watchdog Device Structure
```c
#include <linux/watchdog.h>

struct milspec_watchdog {
    struct watchdog_device wdd;
    void __iomem *base;
    struct clk *clk;
    unsigned int timeout;
    unsigned int pretimeout;
    bool secure_mode;
    enum wdt_action timeout_action;
    struct work_struct emergency_work;
    atomic_t pet_count;
    ktime_t last_pet;
};

static struct milspec_watchdog milspec_wdt;

/* Watchdog operations */
static const struct watchdog_ops milspec_wdt_ops = {
    .owner = THIS_MODULE,
    .start = milspec_wdt_start,
    .stop = milspec_wdt_stop,
    .ping = milspec_wdt_ping,
    .set_timeout = milspec_wdt_set_timeout,
    .set_pretimeout = milspec_wdt_set_pretimeout,
    .get_timeleft = milspec_wdt_get_timeleft,
};

/* Watchdog info */
static const struct watchdog_info milspec_wdt_info = {
    .options = WDIOF_SETTIMEOUT | WDIOF_KEEPALIVEPING | 
               WDIOF_MAGICCLOSE | WDIOF_PRETIMEOUT,
    .firmware_version = 1,
    .identity = "Dell MIL-SPEC Watchdog",
};
```

#### 2.2 Watchdog Operations Implementation
```c
static int milspec_wdt_start(struct watchdog_device *wdd)
{
    struct milspec_watchdog *wdt = watchdog_get_drvdata(wdd);
    u32 control;
    
    /* Set timeout value */
    milspec_write_reg(MILSPEC_REG_WDT_TIMEOUT, wdt->timeout * 1000);
    
    /* Configure watchdog action based on security level */
    if (milspec_state.mode5_level >= MODE5_PARANOID) {
        control = WDT_CTRL_ENABLE | WDT_CTRL_SECURE | WDT_CTRL_RESET;
        wdt->timeout_action = WDT_ACTION_WIPE;
    } else {
        control = WDT_CTRL_ENABLE | WDT_CTRL_RESET;
        wdt->timeout_action = WDT_ACTION_RESET;
    }
    
    /* Enable cascade with TPM watchdog if available */
    if (tpm_ctx.available) {
        control |= WDT_CTRL_CASCADE;
    }
    
    /* Start watchdog */
    milspec_write_reg(MILSPEC_REG_WDT_CONTROL, control);
    
    /* Schedule periodic pet work */
    schedule_delayed_work(&wdt->pet_work, HZ);
    
    return 0;
}

static int milspec_wdt_ping(struct watchdog_device *wdd)
{
    struct milspec_watchdog *wdt = watchdog_get_drvdata(wdd);
    
    /* Pet the watchdog */
    milspec_write_reg(MILSPEC_REG_WDT_COUNTER, wdt->timeout * 1000);
    
    /* Update statistics */
    atomic_inc(&wdt->pet_count);
    wdt->last_pet = ktime_get();
    
    /* Check system health */
    if (milspec_wdt_health_check() != 0) {
        pr_warn("MIL-SPEC: Watchdog health check failed\n");
        /* Reduce timeout for faster recovery */
        milspec_wdt_set_timeout(wdd, wdt->timeout / 2);
    }
    
    return 0;
}
```

### **Phase 3: Advanced Watchdog Features**

#### 3.1 Multi-Stage Timeout Handling
```c
/* Pretimeout warning handler */
static irqreturn_t milspec_wdt_pretimeout_irq(int irq, void *data)
{
    struct milspec_watchdog *wdt = data;
    
    pr_crit("MIL-SPEC: Watchdog pretimeout - system may be compromised\n");
    
    /* Log to TPM */
    milspec_tpm_measure_event("Watchdog pretimeout");
    
    /* Check if emergency action needed */
    if (milspec_state.mode5_level >= MODE5_ENHANCED) {
        /* Attempt graceful shutdown */
        schedule_work(&wdt->emergency_work);
    }
    
    /* Try to recover */
    milspec_wdt_emergency_recover();
    
    return IRQ_HANDLED;
}

/* Emergency recovery procedures */
static void milspec_wdt_emergency_recover(void)
{
    /* Kill non-essential processes */
    /* Sync filesystems */
    /* Disable network interfaces */
    /* Clear sensitive memory */
    /* Prepare for reset/wipe */
}
```

#### 3.2 Integration with Security Subsystems
```c
/* Health check for watchdog pet */
static int milspec_wdt_health_check(void)
{
    int score = 0;
    
    /* Check intrusion status */
    if (milspec_state.intrusion_detected) {
        score += 50;
    }
    
    /* Check TPM PCR values */
    if (milspec_tpm_verify_integrity() != 0) {
        score += 30;
    }
    
    /* Check DSMIL device states */
    for (int i = 0; i < 10; i++) {
        if (!milspec_state.dsmil_active[i] && dsmil_devices[i].critical) {
            score += 20;
        }
    }
    
    /* Check memory integrity */
    if (milspec_check_memory_corruption() != 0) {
        score += 40;
    }
    
    return score; /* 0 = healthy, >100 = critical */
}
```

### **Phase 4: Secure Watchdog Mode**

#### 4.1 Tamper-Resistant Configuration
```c
/* Secure watchdog initialization */
static int milspec_wdt_init_secure(void)
{
    u32 status;
    
    /* Check if already in secure mode */
    status = milspec_read_reg(MILSPEC_REG_WDT_STATUS);
    if (status & WDT_STATUS_LOCKED) {
        pr_info("MIL-SPEC: Watchdog already in secure mode\n");
        return 0;
    }
    
    /* Set secure configuration */
    milspec_write_reg(MILSPEC_REG_WDT_TIMEOUT, 60000); /* 60 seconds */
    milspec_write_reg(MILSPEC_REG_WDT_CONTROL, 
                      WDT_CTRL_ENABLE | WDT_CTRL_SECURE | 
                      WDT_CTRL_RESET | WDT_CTRL_CASCADE);
    
    /* Lock configuration */
    milspec_write_reg(MILSPEC_REG_WDT_STATUS, WDT_STATUS_LOCK);
    
    /* Verify lock */
    status = milspec_read_reg(MILSPEC_REG_WDT_STATUS);
    if (!(status & WDT_STATUS_LOCKED)) {
        pr_err("MIL-SPEC: Failed to lock watchdog\n");
        return -EIO;
    }
    
    return 0;
}
```

#### 4.2 Emergency Actions
```c
/* Watchdog timeout emergency handler */
static void milspec_wdt_emergency_action(struct work_struct *work)
{
    struct milspec_watchdog *wdt = container_of(work, 
                                                struct milspec_watchdog, 
                                                emergency_work);
    
    pr_crit("MIL-SPEC: WATCHDOG TIMEOUT - EMERGENCY ACTION\n");
    
    switch (wdt->timeout_action) {
    case WDT_ACTION_WARN:
        /* Just log and continue */
        pr_crit("MIL-SPEC: Watchdog timeout warning\n");
        break;
        
    case WDT_ACTION_RESET:
        /* Clean reset */
        pr_crit("MIL-SPEC: Initiating system reset\n");
        emergency_restart();
        break;
        
    case WDT_ACTION_POWEROFF:
        /* Power off */
        pr_crit("MIL-SPEC: Initiating power off\n");
        kernel_power_off();
        break;
        
    case WDT_ACTION_WIPE:
        /* Emergency wipe */
        pr_crit("MIL-SPEC: Initiating emergency wipe\n");
        milspec_state.emergency_wipe_armed = true;
        milspec_emergency_wipe();
        break;
    }
}
```

### **Phase 5: Sysfs and Debugfs Interfaces**

#### 5.1 Sysfs Attributes
```c
static ssize_t wdt_status_show(struct device *dev,
                               struct device_attribute *attr, char *buf)
{
    struct milspec_watchdog *wdt = dev_get_drvdata(dev);
    u32 status, counter;
    
    status = milspec_read_reg(MILSPEC_REG_WDT_STATUS);
    counter = milspec_read_reg(MILSPEC_REG_WDT_COUNTER);
    
    return sprintf(buf,
        "Enabled: %s\n"
        "Timeout: %u seconds\n"
        "Time left: %u seconds\n"
        "Pet count: %u\n"
        "Last pet: %lld ms ago\n"
        "Secure mode: %s\n"
        "Action: %s\n",
        status & WDT_CTRL_ENABLE ? "Yes" : "No",
        wdt->timeout,
        counter / 1000,
        atomic_read(&wdt->pet_count),
        ktime_ms_delta(ktime_get(), wdt->last_pet),
        status & WDT_CTRL_SECURE ? "Yes" : "No",
        wdt_action_str(wdt->timeout_action));
}
static DEVICE_ATTR_RO(wdt_status);

static ssize_t wdt_trigger_store(struct device *dev,
                                struct device_attribute *attr,
                                const char *buf, size_t count)
{
    if (sysfs_streq(buf, "TRIGGER_TIMEOUT")) {
        pr_warn("MIL-SPEC: Manual watchdog timeout triggered\n");
        /* Stop petting the watchdog */
        milspec_wdt.manual_trigger = true;
    }
    return count;
}
static DEVICE_ATTR_WO(wdt_trigger);
```

### **Phase 6: Integration with Existing Features**

#### 6.1 Mode 5 Integration
```c
/* Adjust watchdog based on security mode */
static void milspec_wdt_update_mode5(int new_level)
{
    struct milspec_watchdog *wdt = &milspec_wdt;
    
    switch (new_level) {
    case MODE5_DISABLED:
        /* Normal timeout, reset action */
        wdt->timeout = 300; /* 5 minutes */
        wdt->timeout_action = WDT_ACTION_RESET;
        break;
        
    case MODE5_STANDARD:
        /* Shorter timeout */
        wdt->timeout = 180; /* 3 minutes */
        wdt->timeout_action = WDT_ACTION_RESET;
        break;
        
    case MODE5_ENHANCED:
        /* Even shorter, power off */
        wdt->timeout = 120; /* 2 minutes */
        wdt->timeout_action = WDT_ACTION_POWEROFF;
        break;
        
    case MODE5_PARANOID:
    case MODE5_PARANOID_PLUS:
        /* Very short, emergency wipe */
        wdt->timeout = 60; /* 1 minute */
        wdt->timeout_action = WDT_ACTION_WIPE;
        /* Enable secure mode */
        milspec_wdt_init_secure();
        break;
    }
    
    /* Apply new settings */
    if (wdt->wdd.status & WDOG_ACTIVE) {
        milspec_wdt_set_timeout(&wdt->wdd, wdt->timeout);
    }
}
```

## 📊 **Implementation Priority**

### **High Priority:**
1. Basic watchdog timer with Linux framework
2. Hardware register interface
3. Integration with Mode 5 security levels
4. Emergency action handlers

### **Medium Priority:**
5. Pretimeout and multi-stage handling
6. Health check integration
7. Secure mode implementation
8. Sysfs/debugfs interfaces

### **Low Priority:**
9. Advanced recovery procedures
10. Performance monitoring
11. Cascade with external watchdogs
12. Remote management interface

## ⚠️ **Security Considerations**

1. **Tamper Resistance**: Watchdog must not be disableable in high security modes
2. **Fail Secure**: Default to most secure action on any failure
3. **Audit Trail**: All watchdog events must be logged to TPM
4. **Recovery**: Emergency procedures must complete even if system compromised
5. **Testing**: Extensive testing required for timeout scenarios

## 📅 **Implementation Timeline**

- **Week 1**: Basic watchdog implementation and Linux integration
- **Week 2**: Security mode and emergency actions
- **Week 3**: Integration with existing subsystems
- **Week 4**: Testing and hardening

## 🔧 **Testing Strategy**

1. **Unit Tests**: Each watchdog operation
2. **Integration Tests**: With Mode 5, TPM, intrusion detection
3. **Stress Tests**: Continuous operation under load
4. **Fault Injection**: Simulate hangs and compromises
5. **Hardware Tests**: Verify actual hardware behavior

---

**Status**: Plan Ready for Implementation
**Priority**: High - Critical for system reliability
**Estimated Effort**: 4 weeks development + testing
**Dependencies**: Hardware documentation, test hardware