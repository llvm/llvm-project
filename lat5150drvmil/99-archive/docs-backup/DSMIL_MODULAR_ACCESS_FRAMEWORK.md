# DSMIL Modular Access Framework

**Architecture**: 6-Group Modular Design (DSMIL0-5)  
**Approach**: Group-based abstraction with device-level control  
**Safety**: Built-in validation, rollback, and monitoring  
**Scalability**: Extensible for additional groups/devices  

## 🏗️ **Framework Architecture**

### **Group Abstraction Model**
```c
// Core framework structure
struct dsmil_group {
    int group_id;                    // 0-5
    const char *group_name;          // Human readable name  
    struct dsmil_device devices[12]; // 12 devices per group
    u32 active_mask;                 // Bitmask of active devices
    u32 dependency_mask;             // Required groups for activation
    enum dsmil_group_state state;    // Current group state
    struct dsmil_group_ops *ops;     // Group-specific operations
    void *private_data;              // Group-specific data
};

enum dsmil_group_state {
    DSMIL_GROUP_UNINITIALIZED = 0,
    DSMIL_GROUP_INITIALIZED,
    DSMIL_GROUP_ACTIVATING, 
    DSMIL_GROUP_ACTIVE,
    DSMIL_GROUP_DEACTIVATING,
    DSMIL_GROUP_ERROR,
    DSMIL_GROUP_DISABLED
};
```

### **Device-Level Framework**
```c  
struct dsmil_device {
    int device_id;                   // 0-11 (0x0-0xB)
    int group_id;                    // Parent group 0-5
    const char *device_name;         // Device function name
    u32 capabilities;                // Device capability flags
    u32 dependencies;                // Required devices (bitmask)
    enum dsmil_device_state state;   // Current device state
    struct acpi_device *acpi_dev;    // ACPI device handle
    void __iomem *mmio_base;         // MMIO base address
    struct dsmil_device_ops *ops;    // Device operations
    void *private_data;              // Device-specific data
};

enum dsmil_device_state {
    DSMIL_DEVICE_UNKNOWN = 0,
    DSMIL_DEVICE_PRESENT,
    DSMIL_DEVICE_INITIALIZED,
    DSMIL_DEVICE_READY,
    DSMIL_DEVICE_ACTIVE,
    DSMIL_DEVICE_ERROR,
    DSMIL_DEVICE_DISABLED
};
```

## 🔧 **Group Definitions**

### **Group 0: Core Security** (Foundation Layer)
```c
static const struct dsmil_group_config group0_config = {
    .group_id = 0,
    .group_name = "Core Security",
    .dependency_mask = 0x00,  // No dependencies - foundation layer
    .critical = true,         // Required for system operation
    .devices = {
        {0x0, "Security Controller",  DSMIL_CAP_CRITICAL | DSMIL_CAP_FOUNDATION},
        {0x1, "Crypto Engine",        DSMIL_CAP_CRITICAL | DSMIL_CAP_CRYPTO},
        {0x2, "Secure Storage",       DSMIL_CAP_CRITICAL | DSMIL_CAP_STORAGE},
        {0x3, "Network Filter",       DSMIL_CAP_OPTIONAL | DSMIL_CAP_NETWORK},
        {0x4, "Audit Logger",         DSMIL_CAP_OPTIONAL | DSMIL_CAP_LOGGING},
        {0x5, "TPM Interface",        DSMIL_CAP_OPTIONAL | DSMIL_CAP_CRYPTO},
        {0x6, "Secure Boot",          DSMIL_CAP_OPTIONAL | DSMIL_CAP_BOOT},
        {0x7, "Memory Protection",    DSMIL_CAP_OPTIONAL | DSMIL_CAP_MEMORY},
        {0x8, "Tactical Comms",       DSMIL_CAP_CLASSIFIED | DSMIL_CAP_COMMS},
        {0x9, "Emergency Wipe",       DSMIL_CAP_CRITICAL | DSMIL_CAP_SECURITY},
        {0xA, "JROTC Training",       DSMIL_CAP_EDUCATIONAL | DSMIL_CAP_TRAINING},
        {0xB, "Hidden Operations",    DSMIL_CAP_CLASSIFIED | DSMIL_CAP_MEMORY}
    }
};
```

### **Group 1: Extended Security** (Enhancement Layer)
```c
static const struct dsmil_group_config group1_config = {
    .group_id = 1,
    .group_name = "Extended Security", 
    .dependency_mask = 0x01,  // Requires Group 0 active
    .critical = false,
    .devices = {
        {0x0, "Advanced Crypto",      DSMIL_CAP_ENHANCED | DSMIL_CAP_CRYPTO},
        {0x1, "Key Management",       DSMIL_CAP_ENHANCED | DSMIL_CAP_CRYPTO}, 
        {0x2, "Secure Networking",    DSMIL_CAP_ENHANCED | DSMIL_CAP_NETWORK},
        {0x3, "Identity Verification", DSMIL_CAP_ENHANCED | DSMIL_CAP_AUTH},
        {0x4, "Intrusion Detection",  DSMIL_CAP_ENHANCED | DSMIL_CAP_SECURITY},
        {0x5, "Forensic Analysis",    DSMIL_CAP_ENHANCED | DSMIL_CAP_LOGGING},
        {0x6, "Secure Comms Gateway", DSMIL_CAP_ENHANCED | DSMIL_CAP_COMMS},
        {0x7, "Data Classification",  DSMIL_CAP_ENHANCED | DSMIL_CAP_DATA},
        {0x8, "Advanced Tactical",    DSMIL_CAP_CLASSIFIED | DSMIL_CAP_COMMS},
        {0x9, "Secure Backup",        DSMIL_CAP_ENHANCED | DSMIL_CAP_STORAGE},
        {0xA, "Advanced Training",    DSMIL_CAP_EDUCATIONAL | DSMIL_CAP_TRAINING},
        {0xB, "Extended Hidden Ops",  DSMIL_CAP_CLASSIFIED | DSMIL_CAP_MEMORY}
    }
};
```

### **Groups 2-5: Specialized Functions** (Application Layers)
```c
// Group 2: Network Operations
static const struct dsmil_group_config group2_config = {
    .group_id = 2,
    .group_name = "Network Operations",
    .dependency_mask = 0x03,  // Requires Groups 0,1
    .specialized = true
};

// Group 3: Data Processing  
static const struct dsmil_group_config group3_config = {
    .group_id = 3,
    .group_name = "Data Processing", 
    .dependency_mask = 0x01,  // Requires Group 0
    .specialized = true
};

// Group 4: Communications
static const struct dsmil_group_config group4_config = {
    .group_id = 4,
    .group_name = "Communications",
    .dependency_mask = 0x07,  // Requires Groups 0,1,2
    .specialized = true
};

// Group 5: Advanced Features
static const struct dsmil_group_config group5_config = {
    .group_id = 5,  
    .group_name = "Advanced Features",
    .dependency_mask = 0x1F,  // Requires Groups 0-4
    .specialized = true
};
```

## 🔌 **Operation Interfaces**

### **Group Operations Structure**
```c
struct dsmil_group_ops {
    // Group lifecycle
    int (*probe)(struct dsmil_group *group);
    int (*remove)(struct dsmil_group *group);
    int (*suspend)(struct dsmil_group *group);
    int (*resume)(struct dsmil_group *group);
    
    // Group control
    int (*activate)(struct dsmil_group *group, u32 device_mask);
    int (*deactivate)(struct dsmil_group *group, u32 device_mask);
    int (*reset)(struct dsmil_group *group);
    
    // Group status  
    int (*get_status)(struct dsmil_group *group, struct dsmil_group_status *status);
    int (*get_capabilities)(struct dsmil_group *group, u32 *capabilities);
    
    // Group coordination
    int (*check_dependencies)(struct dsmil_group *group);
    int (*validate_config)(struct dsmil_group *group);
    
    // Error handling
    int (*handle_error)(struct dsmil_group *group, int error_code);
    int (*emergency_shutdown)(struct dsmil_group *group);
};
```

### **Device Operations Structure**  
```c
struct dsmil_device_ops {
    // Device lifecycle
    int (*probe)(struct dsmil_device *device);
    int (*remove)(struct dsmil_device *device);
    
    // Device control
    int (*activate)(struct dsmil_device *device);
    int (*deactivate)(struct dsmil_device *device);
    int (*configure)(struct dsmil_device *device, void *config);
    
    // Device I/O
    ssize_t (*read)(struct dsmil_device *device, char *buffer, size_t size);
    ssize_t (*write)(struct dsmil_device *device, const char *buffer, size_t size);
    long (*ioctl)(struct dsmil_device *device, unsigned int cmd, unsigned long arg);
    
    // Device status
    int (*get_status)(struct dsmil_device *device, struct dsmil_device_status *status);
    int (*self_test)(struct dsmil_device *device);
    
    // Error handling  
    int (*handle_error)(struct dsmil_device *device, int error_code);
};
```

## 📋 **Framework Implementation**

### **Core Framework Manager**
```c
// dsmil_framework.c
static struct dsmil_group *dsmil_groups[6];
static DEFINE_MUTEX(dsmil_framework_mutex);

int dsmil_framework_init(void)
{
    int ret, i;
    
    pr_info("DSMIL: Initializing modular framework\n");
    
    mutex_lock(&dsmil_framework_mutex);
    
    // Initialize all groups
    for (i = 0; i < 6; i++) {
        ret = dsmil_group_init(i);
        if (ret) {
            pr_err("DSMIL: Failed to init group %d: %d\n", i, ret);
            goto rollback;
        }
    }
    
    // Validate inter-group dependencies
    ret = dsmil_validate_group_dependencies();
    if (ret) {
        pr_err("DSMIL: Group dependency validation failed: %d\n", ret);
        goto rollback;
    }
    
    mutex_unlock(&dsmil_framework_mutex);
    pr_info("DSMIL: Framework initialization complete\n");
    return 0;
    
rollback:
    // Cleanup initialized groups
    while (--i >= 0)
        dsmil_group_cleanup(i);
    mutex_unlock(&dsmil_framework_mutex);
    return ret;
}

int dsmil_group_activate(int group_id, u32 device_mask)
{
    struct dsmil_group *group;
    int ret;
    
    if (group_id < 0 || group_id >= 6)
        return -EINVAL;
        
    mutex_lock(&dsmil_framework_mutex);
    
    group = dsmil_groups[group_id];
    if (!group) {
        ret = -ENODEV;
        goto out;
    }
    
    // Check dependencies
    ret = group->ops->check_dependencies(group);
    if (ret) {
        pr_err("DSMIL: Group %d dependency check failed: %d\n", group_id, ret);
        goto out;
    }
    
    // Activate requested devices
    ret = group->ops->activate(group, device_mask);
    if (ret) {
        pr_err("DSMIL: Group %d activation failed: %d\n", group_id, ret);
        // Attempt rollback
        group->ops->deactivate(group, group->active_mask);
        goto out;
    }
    
    group->active_mask |= device_mask;
    group->state = DSMIL_GROUP_ACTIVE;
    
    pr_info("DSMIL: Group %d activated (mask=0x%x)\n", group_id, device_mask);
    
out:
    mutex_unlock(&dsmil_framework_mutex);
    return ret;
}
```

### **Safe Activation Sequences**
```c
// dsmil_sequences.c - Predefined safe activation patterns

// Basic activation: Core security only
int dsmil_activate_basic_security(void)
{
    int ret;
    
    pr_info("DSMIL: Activating basic security configuration\n");
    
    // Group 0: Core devices only (0,1,2,9)
    ret = dsmil_group_activate(0, 0x20B);  // Devices 0,1,2,9  
    if (ret) {
        pr_err("DSMIL: Basic security activation failed: %d\n", ret);
        return ret;
    }
    
    // Verify activation
    if (!dsmil_group_is_stable(0, 30)) {  // Wait 30 seconds for stability
        pr_err("DSMIL: Basic security not stable after activation\n");
        dsmil_group_deactivate(0, 0x20B);
        return -ETIMEDOUT;
    }
    
    pr_info("DSMIL: Basic security activated successfully\n");
    return 0;
}

// Enhanced activation: Core + extended features  
int dsmil_activate_enhanced_security(void)
{
    int ret;
    
    pr_info("DSMIL: Activating enhanced security configuration\n");
    
    // First activate basic security
    ret = dsmil_activate_basic_security();
    if (ret)
        return ret;
        
    // Add enhanced devices from Group 0
    ret = dsmil_group_activate(0, 0x1F0);  // Devices 4,5,6,7,8
    if (ret) {
        pr_err("DSMIL: Enhanced security activation failed: %d\n", ret);
        dsmil_group_deactivate(0, 0x3FB);  // Rollback all
        return ret;
    }
    
    // Activate Group 1 basic devices
    ret = dsmil_group_activate(1, 0x00F);  // Devices 0,1,2,3
    if (ret) {
        pr_err("DSMIL: Group 1 activation failed: %d\n", ret);
        dsmil_group_deactivate(0, 0x3FB);  // Rollback Group 0
        return ret;
    }
    
    pr_info("DSMIL: Enhanced security activated successfully\n");
    return 0;
}

// Progressive activation: Carefully staged multi-group
int dsmil_activate_progressive(void)
{
    struct dsmil_activation_plan plan[] = {
        {0, 0x00B, "Group 0 critical devices"},      // 0,1,9
        {0, 0x004, "Group 0 secure storage"},        // 2  
        {0, 0x010, "Group 0 audit logging"},         // 4
        {0, 0x020, "Group 0 TPM interface"},         // 5
        {0, 0x040, "Group 0 secure boot"},           // 6
        {0, 0x080, "Group 0 memory protection"},     // 7
        {1, 0x00F, "Group 1 basic features"},        // 0,1,2,3
        {1, 0x0F0, "Group 1 advanced features"},     // 4,5,6,7
        {2, 0x003, "Group 2 network basics"},        // 0,1
        // ... continue as validation succeeds
    };
    
    return dsmil_execute_activation_plan(plan, ARRAY_SIZE(plan));
}
```

## 🛡️ **Safety and Validation Framework**

### **Pre-Activation Validation**
```c  
int dsmil_validate_activation_safety(int group_id, u32 device_mask)
{
    struct dsmil_group *group = dsmil_groups[group_id];
    int ret, i;
    
    // 1. Dependency validation
    ret = dsmil_check_group_dependencies(group_id);
    if (ret) {
        pr_err("DSMIL: Group %d dependencies not met\n", group_id);
        return ret;
    }
    
    // 2. System resource validation  
    ret = dsmil_check_system_resources(group_id, device_mask);
    if (ret) {
        pr_err("DSMIL: Insufficient system resources\n");  
        return ret;
    }
    
    // 3. Device readiness check
    for_each_set_bit(i, (unsigned long *)&device_mask, 12) {
        ret = group->devices[i].ops->self_test(&group->devices[i]);
        if (ret) {
            pr_err("DSMIL: Device %d.%d failed self-test: %d\n", 
                   group_id, i, ret);
            return ret;
        }
    }
    
    // 4. Thermal safety check
    ret = dsmil_check_thermal_safety();
    if (ret) {
        pr_err("DSMIL: System too hot for activation\n");
        return ret;
    }
    
    return 0;
}
```

### **Rollback Framework**
```c
int dsmil_emergency_rollback(void)
{
    int i, ret = 0;
    
    pr_alert("DSMIL: EMERGENCY ROLLBACK INITIATED\n");
    
    // Deactivate all groups in reverse order
    for (i = 5; i >= 0; i--) {
        if (dsmil_groups[i] && dsmil_groups[i]->state == DSMIL_GROUP_ACTIVE) {
            ret = dsmil_groups[i]->ops->emergency_shutdown(dsmil_groups[i]);
            if (ret) {
                pr_alert("DSMIL: Emergency shutdown group %d failed: %d\n", i, ret);
            } else {
                pr_alert("DSMIL: Group %d emergency shutdown complete\n", i);
            }
        }
    }
    
    // System stabilization delay
    msleep(5000);
    
    pr_alert("DSMIL: Emergency rollback complete\n");
    return ret;
}
```

## 📊 **Framework Status and Monitoring**

### **Status Reporting Interface**
```c
struct dsmil_framework_status {
    u32 framework_version;
    u32 groups_initialized;      // Bitmask of initialized groups
    u32 groups_active;           // Bitmask of active groups  
    u32 total_devices_active;    // Total active devices across all groups
    u32 error_count;             // Total framework errors
    u64 uptime_ns;              // Framework uptime
    struct dsmil_group_status group_status[6];
};

int dsmil_get_framework_status(struct dsmil_framework_status *status)
{
    int i;
    
    mutex_lock(&dsmil_framework_mutex);
    
    status->framework_version = DSMIL_FRAMEWORK_VERSION;
    status->groups_initialized = 0;
    status->groups_active = 0;
    status->total_devices_active = 0;
    
    for (i = 0; i < 6; i++) {
        if (dsmil_groups[i]) {
            status->groups_initialized |= BIT(i);
            
            if (dsmil_groups[i]->state == DSMIL_GROUP_ACTIVE) {
                status->groups_active |= BIT(i);
                status->total_devices_active += hweight32(dsmil_groups[i]->active_mask);
            }
            
            dsmil_groups[i]->ops->get_status(dsmil_groups[i], &status->group_status[i]);
        }
    }
    
    mutex_unlock(&dsmil_framework_mutex);
    return 0;
}
```

---

**Status**: Modular access framework complete - provides safe, scalable group-based abstraction for 72 DSMIL devices with comprehensive safety, validation, and rollback capabilities.