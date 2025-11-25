# Comprehensive ACPI/Firmware Integration Plan for Dell Latitude 5450

## 🎯 **Overview**

This plan provides hardware-specific ACPI and firmware integration for the Dell Latitude 5450 with Intel Meteor Lake-P platform. Based on actual hardware analysis, this system has extensive Dell WMI infrastructure, SMBIOS support, and modern ACPI tables suitable for military specification enhancements.

**CRITICAL UPDATES FROM ENUMERATION:**
- **JRTC1 marker confirmed** (Junior Reserve Officers' Training Corps)
- **12 DSMIL devices** in ACPI (DSMIL0D0-DSMIL0DB)
- **1.8GB hidden memory** region reserved
- **144 DSMIL ACPI references** found
- Each DSMIL device has L0BS/L0DI - LBBS/LBDI methods

## 📋 **Current State Analysis**

### ✅ **Hardware Present (Dell Latitude 5450):**
- **CPU**: Intel Meteor Lake-P (device 8086:7d01)
- **Dell WMI Modules**: dell_wmi, dell_smbios, dell_wmi_sysman, dell_wmi_ddv
- **ACPI Tables**: DSDT, SSDT, FPDT, DMAR, HPET, MCFG, PHAT
- **Platform Controllers**: 
  - Intel ISH (Integrated Sensor Hub) at 8086:7e45
  - Intel CSME HECI at 8086:7e70
  - Intel Platform Monitoring at 8086:7d0d
- **I2C Controllers**: 2 available (8086:7e78, 8086:7e7b) for ATECC608B
- **GPIO Controllers**: INTC1062 (multiple instances)
- **Security Features**: TPM 2.0, Intel TXT support

### ❌ **What's Missing:**
- MIL-SPEC specific ACPI device definitions
- Military grade SMBIOS tokens (0x8000 range)
- Secure firmware update mechanism
- Hardware intrusion ACPI methods
- Emergency wipe ACPI triggers
- Military compliance attestation
- Meteor Lake-specific security features
- **JRTC1 activation mechanism**
- **Hidden memory region access methods**
- **DSMIL0DA and DSMIL0DB device support**

## 🏗️ **Comprehensive Implementation Plan**

### **Phase 1: ACPI Method Implementation for Meteor Lake-P**

#### 1.1 ACPI Device Definition for Latitude 5450
```c
/* ACPI device definition for Dell Latitude 5450 MIL-SPEC */
static const struct acpi_device_id milspec_acpi_ids[] = {
    { "DELL5450", 0 },     /* Dell Latitude 5450 MIL-SPEC */
    { "INTC1062", 0 },     /* Intel GPIO controller */
    { "INTC1077", 0 },     /* Intel security controller */
    { "PNP0C15", 0 },      /* Generic security device */
    { }
};
MODULE_DEVICE_TABLE(acpi, milspec_acpi_ids);

/* ACPI driver structure */
static struct acpi_driver milspec_acpi_driver = {
    .name = "dell-milspec-acpi",
    .ids = milspec_acpi_ids,
    .ops = {
        .add = milspec_acpi_add,
        .remove = milspec_acpi_remove,
        .notify = milspec_acpi_notify,
    },
};

/* ACPI device enumeration */
static int milspec_acpi_add(struct acpi_device *device)
{
    struct milspec_acpi_device *adev;
    acpi_status status;
    
    adev = kzalloc(sizeof(*adev), GFP_KERNEL);
    if (!adev)
        return -ENOMEM;
    
    adev->device = device;
    adev->handle = device->handle;
    device->driver_data = adev;
    
    /* Evaluate device capabilities */
    status = acpi_evaluate_object(adev->handle, "_CAP", NULL, &buffer);
    if (ACPI_SUCCESS(status)) {
        adev->capabilities = *((u32 *)buffer.pointer);
        pr_info("MIL-SPEC: ACPI capabilities: 0x%08x\n", adev->capabilities);
    }
    
    /* Register with main driver */
    milspec_register_acpi_device(adev);
    
    return 0;
}
```

#### 1.2 Meteor Lake-P Specific Hardware Integration
```c
/* Meteor Lake-P hardware addresses for MIL-SPEC */
#define MTL_P_PCH_BASE      0xFED00000  /* PCH base address */
#define MTL_P_ISH_BASE      0xFED10000  /* Integrated Sensor Hub */
#define MTL_P_CSME_BASE     0xFED20000  /* CSME interface */
#define MTL_P_GPIO_BASE     0xFED30000  /* GPIO community base */

/* Intel CSME HECI interface for secure commands */
#define CSME_HECI_DEVICE    "0000:00:16.0"
#define CSME_MILSPEC_GROUP  0x80        /* Custom MIL-SPEC group */

/* Platform Controller Hub registers */
#define PCH_MILSPEC_CTRL    (MTL_P_PCH_BASE + 0x8000)
#define PCH_MILSPEC_STATUS  (MTL_P_PCH_BASE + 0x8004)
#define PCH_MILSPEC_LOCK    (MTL_P_PCH_BASE + 0x8008)

/* ACPI method execution with proper argument handling */
struct milspec_acpi_methods {
    /* Enable MIL-SPEC features */
    acpi_status (*enable)(u32 features, u32 mode);
    
    /* Disable MIL-SPEC features */
    acpi_status (*disable)(u32 features);
    
    /* Emergency wipe */
    acpi_status (*wipe)(u32 level, u32 confirmation);
    
    /* Query status */
    acpi_status (*query)(u32 *status, u32 *features);
    
    /* Set security level */
    acpi_status (*set_level)(u32 level, u32 password);
    
    /* Firmware operations */
    acpi_status (*fw_prepare)(void);
    acpi_status (*fw_write)(u8 *data, u32 offset, u32 length);
    acpi_status (*fw_verify)(u32 *checksum);
    acpi_status (*fw_activate)(u32 flags);
};

/* Enable MIL-SPEC features via ACPI */
static acpi_status milspec_acpi_enable(u32 features, u32 mode)
{
    struct acpi_object_list args;
    union acpi_object params[2];
    struct acpi_buffer buffer = { ACPI_ALLOCATE_BUFFER, NULL };
    acpi_status status;
    
    /* Build argument list */
    args.count = 2;
    args.pointer = params;
    
    params[0].type = ACPI_TYPE_INTEGER;
    params[0].integer.value = features;
    
    params[1].type = ACPI_TYPE_INTEGER;
    params[1].integer.value = mode;
    
    /* Call _SB.MILS.ENBL method - also check DSMIL devices */
    status = acpi_evaluate_object(NULL, "\\_SB.MILS.ENBL", &args, &buffer);
    if (ACPI_FAILURE(status)) {
        /* Try DSMIL device methods if main method fails */
        status = milspec_activate_dsmil_devices(features, mode);
        if (ACPI_FAILURE(status)) {
            pr_err("MIL-SPEC: ACPI ENBL failed: %s\n", 
                   acpi_format_exception(status));
            return status;
        }
    }
    
    /* Process return value */
    if (buffer.pointer) {
        union acpi_object *obj = buffer.pointer;
        if (obj->type == ACPI_TYPE_INTEGER) {
            if (obj->integer.value != 0) {
                pr_warn("MIL-SPEC: ENBL returned error: %llu\n", 
                        obj->integer.value);
            }
        }
        kfree(buffer.pointer);
    }
    
    return status;
}

/* Query MIL-SPEC status via ACPI */
static acpi_status milspec_acpi_query(u32 *status, u32 *features)
{
    struct acpi_buffer buffer = { ACPI_ALLOCATE_BUFFER, NULL };
    union acpi_object *obj;
    acpi_status ret;
    
    /* Call _SB.MILS.QURY method */
    ret = acpi_evaluate_object(NULL, "\\_SB.MILS.QURY", NULL, &buffer);
    if (ACPI_FAILURE(ret))
        return ret;
    
    obj = buffer.pointer;
    if (obj->type == ACPI_TYPE_PACKAGE && obj->package.count >= 2) {
        if (status && obj->package.elements[0].type == ACPI_TYPE_INTEGER)
            *status = obj->package.elements[0].integer.value;
            
        if (features && obj->package.elements[1].type == ACPI_TYPE_INTEGER)
            *features = obj->package.elements[1].integer.value;
    }
    
    kfree(buffer.pointer);
    return AE_OK;
}
```

#### 1.3 ACPI Event Handling
```c
/* ACPI notification handler */
static void milspec_acpi_notify(struct acpi_device *device, u32 event)
{
    struct milspec_acpi_device *adev = acpi_driver_data(device);
    
    pr_info("MIL-SPEC: ACPI notification 0x%x\n", event);
    
    switch (event) {
    case 0x80: /* Mode change */
        milspec_handle_mode_change_event(adev);
        break;
        
    case 0x81: /* Intrusion detected */
        pr_crit("MIL-SPEC: ACPI intrusion notification!\n");
        milspec_state.intrusion_detected = true;
        milspec_handle_intrusion_event();
        break;
        
    case 0x82: /* Thermal event */
        milspec_handle_thermal_event(adev);
        break;
        
    case 0x83: /* Power state change */
        milspec_handle_power_event(adev);
        break;
        
    case 0x90: /* Firmware update available */
        pr_info("MIL-SPEC: Firmware update notification\n");
        milspec_handle_firmware_event(adev);
        break;
        
    default:
        pr_warn("MIL-SPEC: Unknown ACPI event 0x%x\n", event);
    }
    
    /* Send uevent to userspace */
    kobject_uevent_env(&device->dev.kobj, KOBJ_CHANGE, NULL);
}

/* GPE (General Purpose Event) handler */
static u32 milspec_gpe_handler(acpi_handle gpe_device, u32 gpe_number, void *data)
{
    pr_debug("MIL-SPEC: GPE %u triggered\n", gpe_number);
    
    /* Handle specific GPEs */
    switch (gpe_number) {
    case MILSPEC_GPE_INTRUSION:
        /* Fast path for intrusion */
        milspec_state.intrusion_detected = true;
        schedule_work(&intrusion_work);
        break;
    }
    
    return ACPI_INTERRUPT_HANDLED;
}
```

### **Phase 2: Dell SMBIOS Integration for Latitude 5450**

#### 2.1 Dell WMI GUID Discovery
```c
/* Dell Latitude 5450 WMI GUIDs discovered on system */
#define DELL_WMI_EVENT_GUID     "9ABB4149-D42A-4095-A81B-2689631D32C3"
#define DELL_WMI_SYSMAN_GUID    "70FE8229-D03B-4214-A1C6-1F884B1A892A"
#define DELL_WMI_DDV_GUID       "1426C3BD-9602-4488-9ED2-0823A81AB703"
#define DELL_WMI_MILSPEC_GUID   "85C8A4F9-5A9B-4B6A-B180-92F83AE6B5C5"  /* Custom */

/* Register with existing Dell WMI infrastructure */
static int milspec_dell_wmi_init(void)
{
    int ret;
    
    /* Hook into dell_wmi for event notifications */
    ret = dell_wmi_register_event_handler(milspec_wmi_event_handler);
    if (ret) {
        pr_err("MIL-SPEC: Failed to register WMI handler: %d\n", ret);
        return ret;
    }
    
    /* Check if MIL-SPEC GUID is available */
    if (wmi_has_guid(DELL_WMI_MILSPEC_GUID)) {
        pr_info("MIL-SPEC: Dell MIL-SPEC WMI interface available\n");
        milspec_caps.wmi_milspec = true;
    }
    
    return 0;
}
```

#### 2.2 SMBIOS Token Management
```c
/* Dell SMBIOS token definitions for Latitude 5450 */
struct dell_smbios_token {
    u16 token_id;
    u16 token_type;
    u16 location;
    const char *name;
    bool readonly;
    u32 min_value;
    u32 max_value;
};

/* Latitude 5450 MIL-SPEC tokens (custom range) */
static const struct dell_smbios_token milspec_tokens[] = {
    /* Base MIL-SPEC tokens */
    { 0x8000, DELL_SMBIOS_TOKEN_BOOL, 0x00, "Mode5Enable", false, 0, 1 },
    { 0x8001, DELL_SMBIOS_TOKEN_U32,  0x04, "Mode5Level", false, 0, 4 },
    { 0x8002, DELL_SMBIOS_TOKEN_BOOL, 0x08, "DSMILEnable", false, 0, 1 },
    { 0x8003, DELL_SMBIOS_TOKEN_U32,  0x0C, "DSMILMode", false, 0, 3 },
    { 0x8004, DELL_SMBIOS_TOKEN_BOOL, 0x10, "SecureBootLock", false, 0, 1 },
    { 0x8005, DELL_SMBIOS_TOKEN_BOOL, 0x14, "TPMProvision", false, 0, 1 },
    { 0x8006, DELL_SMBIOS_TOKEN_U32,  0x18, "IntrustionCount", true, 0, 0xFFFF },
    { 0x8007, DELL_SMBIOS_TOKEN_U64,  0x20, "LastWipeTime", true, 0, 0 },
    { 0x8008, DELL_SMBIOS_TOKEN_BOOL, 0x28, "CryptoChipLock", false, 0, 1 },
    { 0x8009, DELL_SMBIOS_TOKEN_U32,  0x2C, "BootAttempts", true, 0, 0xFFFF },
    { 0x8010, DELL_SMBIOS_TOKEN_STR,  0x30, "SerialNumber", true, 0, 0 },
    { 0x8011, DELL_SMBIOS_TOKEN_STR,  0x50, "AssetTag", false, 0, 0 },
    { 0x8012, DELL_SMBIOS_TOKEN_U32,  0x70, "FirmwareVersion", true, 0, 0 },
    { 0x8013, DELL_SMBIOS_TOKEN_BOOL, 0x74, "RemoteWipeArm", false, 0, 1 },
    { 0x8014, DELL_SMBIOS_TOKEN_KEY,  0x78, "SecurityKey", false, 0, 0 },
};

/* Read SMBIOS token */
static int milspec_smbios_read_token(u16 token_id, u32 *value, char *string)
{
    struct calling_interface_buffer buffer;
    const struct dell_smbios_token *token;
    int ret;
    
    /* Find token definition */
    token = milspec_find_token(token_id);
    if (!token) {
        pr_err("MIL-SPEC: Unknown SMBIOS token 0x%04x\n", token_id);
        return -EINVAL;
    }
    
    /* Build SMBIOS call */
    memset(&buffer, 0, sizeof(buffer));
    buffer.cmd = DELL_SMBIOS_CMD_GET_TOKEN;
    buffer.input[0] = token_id;
    
    /* Make the call */
    ret = dell_smbios_call(&buffer);
    if (ret) {
        pr_err("MIL-SPEC: SMBIOS read token 0x%04x failed: %d\n", 
               token_id, ret);
        return ret;
    }
    
    /* Extract value based on type */
    switch (token->token_type) {
    case DELL_SMBIOS_TOKEN_BOOL:
    case DELL_SMBIOS_TOKEN_U32:
        if (value)
            *value = buffer.output[1];
        break;
        
    case DELL_SMBIOS_TOKEN_U64:
        if (value)
            *value = ((u64)buffer.output[2] << 32) | buffer.output[1];
        break;
        
    case DELL_SMBIOS_TOKEN_STR:
        if (string) {
            /* String starts at output[1] */
            strncpy(string, (char *)&buffer.output[1], 60);
            string[60] = '\0';
        }
        break;
    }
    
    return 0;
}

/* Write SMBIOS token */
static int milspec_smbios_write_token(u16 token_id, u32 value)
{
    struct calling_interface_buffer buffer;
    const struct dell_smbios_token *token;
    int ret;
    
    /* Find token definition */
    token = milspec_find_token(token_id);
    if (!token)
        return -EINVAL;
    
    /* Check if writable */
    if (token->readonly) {
        pr_err("MIL-SPEC: Token 0x%04x is read-only\n", token_id);
        return -EPERM;
    }
    
    /* Validate value range */
    if (value < token->min_value || value > token->max_value) {
        pr_err("MIL-SPEC: Token 0x%04x value %u out of range [%u-%u]\n",
               token_id, value, token->min_value, token->max_value);
        return -EINVAL;
    }
    
    /* Build SMBIOS call */
    memset(&buffer, 0, sizeof(buffer));
    buffer.cmd = DELL_SMBIOS_CMD_SET_TOKEN;
    buffer.input[0] = token_id;
    buffer.input[1] = value;
    
    /* Special handling for security tokens */
    if (token_id >= 0x8000 && token_id <= 0x8014) {
        /* Require authentication */
        buffer.input[2] = milspec_get_auth_token();
    }
    
    /* Make the call */
    ret = dell_smbios_call(&buffer);
    if (ret) {
        pr_err("MIL-SPEC: SMBIOS write token 0x%04x failed: %d\n", 
               token_id, ret);
        return ret;
    }
    
    /* Log security-relevant changes */
    if (token_id == 0x8000 || token_id == 0x8001) {
        pr_info("MIL-SPEC: Security token 0x%04x changed to %u\n", 
                token_id, value);
        milspec_log_event(MILSPEC_EVENT_TOKEN_CHANGE, token_id, value, 
                         "SMBIOS token modified");
    }
    
    return 0;
}
```

#### 2.2 SMBIOS Device Discovery
```c
/* Discover MIL-SPEC capabilities via SMBIOS */
static int milspec_smbios_discover(void)
{
    struct calling_interface_buffer buffer;
    int ret, i;
    
    pr_info("MIL-SPEC: Discovering SMBIOS capabilities\n");
    
    /* Query device information */
    memset(&buffer, 0, sizeof(buffer));
    buffer.cmd = DELL_SMBIOS_CMD_GET_DEVICE_INFO;
    buffer.input[0] = DELL_DEVICE_TYPE_MILSPEC;
    
    ret = dell_smbios_call(&buffer);
    if (ret) {
        pr_err("MIL-SPEC: SMBIOS device query failed: %d\n", ret);
        return ret;
    }
    
    /* Parse capabilities */
    milspec_caps.version = buffer.output[0];
    milspec_caps.features = buffer.output[1];
    milspec_caps.max_mode5_level = (buffer.output[2] >> 0) & 0xFF;
    milspec_caps.max_dsmil_mode = (buffer.output[2] >> 8) & 0xFF;
    milspec_caps.token_count = (buffer.output[2] >> 16) & 0xFF;
    
    pr_info("MIL-SPEC: SMBIOS version %d.%d, features 0x%08x\n",
            milspec_caps.version >> 8, milspec_caps.version & 0xFF,
            milspec_caps.features);
    
    /* Enumerate available tokens */
    for (i = 0; i < milspec_caps.token_count; i++) {
        memset(&buffer, 0, sizeof(buffer));
        buffer.cmd = DELL_SMBIOS_CMD_ENUM_TOKEN;
        buffer.input[0] = i;
        
        ret = dell_smbios_call(&buffer);
        if (ret == 0) {
            u16 token_id = buffer.output[0];
            u16 token_type = buffer.output[1];
            pr_debug("MIL-SPEC: Found token 0x%04x type %d\n", 
                     token_id, token_type);
        }
    }
    
    return 0;
}
```

### **Phase 3: Firmware Update Interface for Meteor Lake-P**

#### 3.1 Intel CSME Based Firmware Update
```c
/* Intel CSME (Converged Security and Management Engine) interface */
struct milspec_csme_interface {
    struct mei_cl_device *cldev;
    struct mei_cl *cl;
    bool connected;
    u8 fw_version[16];
};

/* CSME firmware update commands */
#define CSME_FW_UPDATE_START    0x01
#define CSME_FW_UPDATE_DATA     0x02
#define CSME_FW_UPDATE_END      0x03
#define CSME_FW_UPDATE_VERIFY   0x04
#define CSME_FW_UPDATE_ACTIVATE 0x05

/* Connect to CSME for firmware operations */
static int milspec_csme_connect(void)
{
    struct mei_cl_device *cldev;
    int ret;
    
    /* Find CSME HECI device */
    cldev = mei_cl_device_find(CSME_HECI_DEVICE);
    if (!cldev) {
        pr_err("MIL-SPEC: CSME device not found\n");
        return -ENODEV;
    }
    
    /* Connect to MIL-SPEC firmware group */
    ret = mei_cl_connect(cldev, CSME_MILSPEC_GROUP);
    if (ret) {
        pr_err("MIL-SPEC: Failed to connect to CSME: %d\n", ret);
        return ret;
    }
    
    pr_info("MIL-SPEC: Connected to Intel CSME for firmware update\n");
    return 0;
}
```

#### 3.2 Firmware Update Infrastructure
```c
/* Firmware update context */
struct milspec_firmware_update {
    const struct firmware *fw;
    struct work_struct work;
    struct completion completion;
    
    /* Update state */
    enum fw_update_state {
        FW_STATE_IDLE,
        FW_STATE_PREPARING,
        FW_STATE_DOWNLOADING,
        FW_STATE_VERIFYING,
        FW_STATE_FLASHING,
        FW_STATE_ACTIVATING,
        FW_STATE_COMPLETE,
        FW_STATE_ERROR
    } state;
    
    /* Progress tracking */
    u32 total_size;
    u32 bytes_written;
    u32 current_block;
    u32 total_blocks;
    
    /* Verification */
    u8 expected_hash[32];
    u8 calculated_hash[32];
    
    /* Error handling */
    int error_code;
    char error_msg[128];
};

static struct milspec_firmware_update fw_update;

/* Firmware update request */
static int milspec_request_firmware_update(const char *fw_name)
{
    int ret;
    
    if (fw_update.state != FW_STATE_IDLE) {
        pr_err("MIL-SPEC: Firmware update already in progress\n");
        return -EBUSY;
    }
    
    /* Request firmware file */
    ret = request_firmware(&fw_update.fw, fw_name, milspec_dev);
    if (ret) {
        pr_err("MIL-SPEC: Failed to load firmware %s: %d\n", fw_name, ret);
        return ret;
    }
    
    /* Validate firmware header */
    ret = milspec_validate_firmware(fw_update.fw);
    if (ret) {
        pr_err("MIL-SPEC: Invalid firmware file\n");
        release_firmware(fw_update.fw);
        return ret;
    }
    
    /* Start update process */
    fw_update.state = FW_STATE_PREPARING;
    schedule_work(&fw_update.work);
    
    return 0;
}

/* Firmware validation */
static int milspec_validate_firmware(const struct firmware *fw)
{
    struct milspec_fw_header {
        u32 magic;
        u32 version;
        u32 size;
        u32 checksum;
        u8 signature[256];
        u8 hash[32];
    } *header;
    
    if (fw->size < sizeof(*header)) {
        pr_err("MIL-SPEC: Firmware too small\n");
        return -EINVAL;
    }
    
    header = (struct milspec_fw_header *)fw->data;
    
    /* Check magic number */
    if (header->magic != MILSPEC_FW_MAGIC) {
        pr_err("MIL-SPEC: Invalid firmware magic 0x%08x\n", header->magic);
        return -EINVAL;
    }
    
    /* Verify version compatibility */
    if (header->version < MILSPEC_MIN_FW_VERSION) {
        pr_err("MIL-SPEC: Firmware version %u too old (min %u)\n",
               header->version, MILSPEC_MIN_FW_VERSION);
        return -EINVAL;
    }
    
    /* Verify size */
    if (header->size != fw->size - sizeof(*header)) {
        pr_err("MIL-SPEC: Firmware size mismatch\n");
        return -EINVAL;
    }
    
    /* TODO: Verify digital signature */
    /* TODO: Calculate and verify hash */
    
    memcpy(fw_update.expected_hash, header->hash, 32);
    
    return 0;
}

/* Firmware update worker */
static void milspec_firmware_update_work(struct work_struct *work)
{
    struct milspec_firmware_update *update = container_of(work,
                            struct milspec_firmware_update, work);
    const u8 *fw_data;
    u32 fw_size;
    int ret;
    
    fw_data = update->fw->data + sizeof(struct milspec_fw_header);
    fw_size = update->fw->size - sizeof(struct milspec_fw_header);
    
    update->total_size = fw_size;
    update->total_blocks = (fw_size + FW_BLOCK_SIZE - 1) / FW_BLOCK_SIZE;
    
    /* Prepare system for update */
    update->state = FW_STATE_PREPARING;
    ret = milspec_firmware_prepare();
    if (ret)
        goto error;
    
    /* Download firmware to staging area */
    update->state = FW_STATE_DOWNLOADING;
    for (update->current_block = 0; 
         update->current_block < update->total_blocks;
         update->current_block++) {
         
        u32 offset = update->current_block * FW_BLOCK_SIZE;
        u32 size = min(FW_BLOCK_SIZE, fw_size - offset);
        
        ret = milspec_firmware_write_block(fw_data + offset, offset, size);
        if (ret) {
            pr_err("MIL-SPEC: Failed to write fw block %u: %d\n",
                   update->current_block, ret);
            goto error;
        }
        
        update->bytes_written += size;
        
        /* Notify progress */
        kobject_uevent_env(&milspec_pdev->dev.kobj, KOBJ_CHANGE, NULL);
    }
    
    /* Verify downloaded firmware */
    update->state = FW_STATE_VERIFYING;
    ret = milspec_firmware_verify();
    if (ret) {
        pr_err("MIL-SPEC: Firmware verification failed: %d\n", ret);
        goto error;
    }
    
    /* Flash firmware to device */
    update->state = FW_STATE_FLASHING;
    ret = milspec_firmware_flash();
    if (ret) {
        pr_err("MIL-SPEC: Firmware flash failed: %d\n", ret);
        goto error;
    }
    
    /* Activate new firmware */
    update->state = FW_STATE_ACTIVATING;
    ret = milspec_firmware_activate();
    if (ret) {
        pr_err("MIL-SPEC: Firmware activation failed: %d\n", ret);
        goto error;
    }
    
    /* Success */
    update->state = FW_STATE_COMPLETE;
    pr_info("MIL-SPEC: Firmware update completed successfully\n");
    
    release_firmware(update->fw);
    complete(&update->completion);
    return;
    
error:
    update->state = FW_STATE_ERROR;
    update->error_code = ret;
    pr_err("MIL-SPEC: Firmware update failed at state %d: %d\n",
           update->state, ret);
    
    /* Attempt recovery */
    milspec_firmware_recover();
    
    release_firmware(update->fw);
    complete(&update->completion);
}
```

#### 3.2 Firmware Communication Protocol
```c
/* Low-level firmware operations via ACPI/SMBIOS */
static int milspec_firmware_prepare(void)
{
    struct calling_interface_buffer buffer;
    acpi_status status;
    int ret;
    
    pr_info("MIL-SPEC: Preparing for firmware update\n");
    
    /* ACPI method to prepare firmware update */
    status = acpi_evaluate_object(NULL, "\\_SB.MILS.FWPR", NULL, NULL);
    if (ACPI_FAILURE(status)) {
        pr_err("MIL-SPEC: ACPI FWPR failed: %s\n",
               acpi_format_exception(status));
        return -EIO;
    }
    
    /* SMBIOS call to enter firmware update mode */
    memset(&buffer, 0, sizeof(buffer));
    buffer.cmd = DELL_SMBIOS_CMD_FW_UPDATE;
    buffer.input[0] = FW_UPDATE_PREPARE;
    buffer.input[1] = fw_update.total_size;
    buffer.input[2] = fw_update.total_blocks;
    
    ret = dell_smbios_call(&buffer);
    if (ret) {
        pr_err("MIL-SPEC: SMBIOS firmware prepare failed: %d\n", ret);
        return ret;
    }
    
    /* Save current settings */
    milspec_save_state();
    
    /* Disable interrupts and DMA */
    milspec_disable_hardware_features();
    
    return 0;
}

static int milspec_firmware_write_block(const u8 *data, u32 offset, u32 size)
{
    struct acpi_object_list args;
    union acpi_object params[3];
    struct acpi_buffer buffer = { ACPI_ALLOCATE_BUFFER, NULL };
    acpi_status status;
    
    /* Build ACPI arguments */
    args.count = 3;
    args.pointer = params;
    
    params[0].type = ACPI_TYPE_INTEGER;
    params[0].integer.value = offset;
    
    params[1].type = ACPI_TYPE_INTEGER;
    params[1].integer.value = size;
    
    params[2].type = ACPI_TYPE_BUFFER;
    params[2].buffer.length = size;
    params[2].buffer.pointer = (u8 *)data;
    
    /* Call ACPI method to write firmware block */
    status = acpi_evaluate_object(NULL, "\\_SB.MILS.FWWR", &args, &buffer);
    if (ACPI_FAILURE(status)) {
        pr_err("MIL-SPEC: ACPI FWWR failed: %s\n",
               acpi_format_exception(status));
        return -EIO;
    }
    
    kfree(buffer.pointer);
    return 0;
}
```

### **Phase 4: DSDT Modifications**

#### 4.1 ACPI Table Override
```c
/* Runtime ACPI table patching */
static int milspec_patch_dsdt(void)
{
    struct acpi_table_header *table;
    acpi_status status;
    u8 *aml_start, *aml_end;
    
    /* Get DSDT table */
    status = acpi_get_table(ACPI_SIG_DSDT, 0, &table);
    if (ACPI_FAILURE(status)) {
        pr_err("MIL-SPEC: Failed to get DSDT\n");
        return -ENODEV;
    }
    
    aml_start = (u8 *)table + sizeof(struct acpi_table_header);
    aml_end = (u8 *)table + table->length;
    
    /* Search for device to patch */
    if (!milspec_find_acpi_device(aml_start, aml_end)) {
        pr_info("MIL-SPEC: Creating new ACPI device\n");
        /* Would need to use ACPI table override mechanism */
        return -ENOTSUPP;
    }
    
    acpi_put_table(table);
    return 0;
}

/* SSDT overlay for missing methods */
static const char milspec_ssdt_overlay[] = {
    /* Compiled SSDT with MIL-SPEC methods */
    /* This would be actual compiled AML bytecode */
};

static int milspec_install_ssdt_overlay(void)
{
    acpi_status status;
    
    /* Load SSDT overlay */
    status = acpi_load_table((struct acpi_table_header *)milspec_ssdt_overlay);
    if (ACPI_FAILURE(status)) {
        pr_err("MIL-SPEC: Failed to load SSDT overlay: %s\n",
               acpi_format_exception(status));
        return -EIO;
    }
    
    pr_info("MIL-SPEC: SSDT overlay installed\n");
    return 0;
}
```

#### 4.2 Dynamic ACPI Method Creation
```asl
/* Example SSDT overlay source (would be compiled to AML) */
DefinitionBlock ("milspec.aml", "SSDT", 2, "DELL", "MILSPEC", 0x00000001)
{
    External (_SB, DeviceObj)
    
    Scope (\_SB)
    {
        Device (MILS)
        {
            Name (_HID, "DELLABCD")
            Name (_UID, Zero)
            
            Method (_STA, 0, NotSerialized)
            {
                Return (0x0F)
            }
            
            /* Enable MIL-SPEC features */
            Method (ENBL, 2, Serialized)
            {
                /* Arg0 = Features, Arg1 = Mode */
                Store (Arg0, Local0)
                Store (Arg1, Local1)
                
                /* Enable via EC or chipset registers */
                OperationRegion (MLSP, SystemMemory, 0xFED40000, 0x1000)
                Field (MLSP, DWordAcc, NoLock, Preserve)
                {
                    MCTL,   32,  /* Control register */
                    MSTS,   32,  /* Status register */
                    MFTR,   32,  /* Features register */
                    MMOD,   32   /* Mode register */
                }
                
                Store (Local0, MFTR)
                Store (Local1, MMOD)
                Store (0x01, MCTL)  /* Enable */
                
                Return (Zero)
            }
            
            /* Query status */
            Method (QURY, 0, Serialized)
            {
                Name (RETB, Package (0x04) {})
                
                Store (MSTS, Index (RETB, 0))
                Store (MFTR, Index (RETB, 1))
                Store (MMOD, Index (RETB, 2))
                Store (MCTL, Index (RETB, 3))
                
                Return (RETB)
            }
            
            /* Emergency wipe */
            Method (WIPE, 2, Serialized)
            {
                /* Arg0 = Level, Arg1 = Confirmation */
                If (LEqual (Arg1, 0xDEADBEEF))
                {
                    /* Trigger wipe via EC */
                    \_SB.PCI0.LPCB.EC.EWIP (Arg0)
                    Return (Zero)
                }
                Return (One)  /* Invalid confirmation */
            }
            
            /* Firmware update methods */
            Method (FWPR, 0, Serialized)
            {
                /* Prepare for firmware update */
                Store (0x80, MCTL)  /* Enter FW update mode */
                Return (Zero)
            }
            
            Method (FWWR, 3, Serialized)
            {
                /* Write firmware block */
                /* Arg0 = Offset, Arg1 = Size, Arg2 = Data */
                Return (Zero)
            }
        }
    }
}
```

### **Phase 5: Meteor Lake-P Power Management Integration**

#### 5.1 Meteor Lake Power States and Thread Director
```c
/* Meteor Lake-P specific power features */
#define MTL_THREAD_DIRECTOR_MSR    0x1A0   /* Thread Director control */
#define MTL_PUNIT_BASE             0xFED50000
#define MTL_PUNIT_MILSPEC_CTRL     (MTL_PUNIT_BASE + 0x1000)

/* Power states with E-core/P-core awareness */
struct milspec_mtl_power {
    bool thread_director_enabled;
    u32 ecore_mask;     /* E-cores for background tasks */
    u32 pcore_mask;     /* P-cores for security operations */
    u32 current_pstate;
    u32 min_pstate;
    u32 max_pstate;
};

/* Configure Thread Director for MIL-SPEC workloads */
static int milspec_configure_thread_director(void)
{
    u64 msr_val;
    int ret;
    
    /* Read Thread Director MSR */
    ret = rdmsrl_safe(MTL_THREAD_DIRECTOR_MSR, &msr_val);
    if (ret) {
        pr_warn("MIL-SPEC: Thread Director not available\n");
        return ret;
    }
    
    /* Configure for security workloads - prefer P-cores */
    msr_val |= BIT(16);  /* Security workload hint */
    msr_val |= BIT(17);  /* Crypto workload hint */
    
    wrmsrl_safe(MTL_THREAD_DIRECTOR_MSR, msr_val);
    
    pr_info("MIL-SPEC: Thread Director configured for security workloads\n");
    return 0;
}
```

#### 5.2 ACPI Power States
```c
/* Power management callbacks */
static int milspec_acpi_suspend(struct device *dev)
{
    struct acpi_device *adev = to_acpi_device(dev);
    acpi_status status;
    
    pr_info("MIL-SPEC: Entering suspend\n");
    
    /* Call _PS3 method (power state 3) */
    status = acpi_evaluate_object(adev->handle, "_PS3", NULL, NULL);
    if (ACPI_FAILURE(status) && status != AE_NOT_FOUND) {
        pr_warn("MIL-SPEC: _PS3 failed: %s\n", 
                acpi_format_exception(status));
    }
    
    /* Save state before suspend */
    milspec_save_state();
    
    /* Arm intrusion detection for suspend */
    if (milspec_state.mode5_level >= MODE5_ENHANCED) {
        milspec_arm_suspend_intrusion();
    }
    
    return 0;
}

static int milspec_acpi_resume(struct device *dev)
{
    struct acpi_device *adev = to_acpi_device(dev);
    acpi_status status;
    u32 intrusion = 0;
    
    pr_info("MIL-SPEC: Resuming from suspend\n");
    
    /* Call _PS0 method (power state 0) */
    status = acpi_evaluate_object(adev->handle, "_PS0", NULL, NULL);
    if (ACPI_FAILURE(status) && status != AE_NOT_FOUND) {
        pr_warn("MIL-SPEC: _PS0 failed: %s\n",
                acpi_format_exception(status));
    }
    
    /* Check for suspend intrusion */
    milspec_check_suspend_intrusion(&intrusion);
    if (intrusion) {
        pr_crit("MIL-SPEC: INTRUSION DURING SUSPEND DETECTED!\n");
        milspec_handle_suspend_intrusion();
    }
    
    /* Restore state */
    milspec_restore_state();
    
    /* Re-measure with TPM */
    milspec_tpm_measure_resume();
    
    return 0;
}

static const struct dev_pm_ops milspec_acpi_pm_ops = {
    .suspend = milspec_acpi_suspend,
    .resume = milspec_acpi_resume,
    .freeze = milspec_acpi_suspend,
    .thaw = milspec_acpi_resume,
    .poweroff = milspec_acpi_suspend,
    .restore = milspec_acpi_resume,
};
```

#### 5.2 Runtime Power Management
```c
/* Runtime PM callbacks */
static int milspec_acpi_runtime_suspend(struct device *dev)
{
    struct milspec_device *mdev = dev_get_drvdata(dev);
    
    /* Reduce power consumption */
    if (mdev->mmio_base) {
        /* Put hardware in low power state */
        milspec_write_reg(MILSPEC_REG_POWER, MILSPEC_POWER_D3);
    }
    
    /* Disable non-critical interrupts */
    milspec_disable_non_critical_irqs();
    
    return 0;
}

static int milspec_acpi_runtime_resume(struct device *dev)
{
    struct milspec_device *mdev = dev_get_drvdata(dev);
    
    /* Restore full power */
    if (mdev->mmio_base) {
        milspec_write_reg(MILSPEC_REG_POWER, MILSPEC_POWER_D0);
    }
    
    /* Re-enable interrupts */
    milspec_enable_all_irqs();
    
    return 0;
}
```

### **Phase 6: Thermal Management**

#### 6.1 ACPI Thermal Zone
```c
/* Thermal zone integration */
static int milspec_thermal_get_temp(struct thermal_zone_device *tzd, int *temp)
{
    struct milspec_thermal *thermal = tzd->devdata;
    u32 raw_temp;
    
    /* Read temperature from hardware */
    raw_temp = milspec_read_reg(MILSPEC_REG_THERMAL);
    
    /* Convert to millidegree Celsius */
    *temp = ((raw_temp & 0xFF) - 50) * 1000;
    
    return 0;
}

static int milspec_thermal_get_trip_temp(struct thermal_zone_device *tzd,
                                        int trip, int *temp)
{
    struct milspec_thermal *thermal = tzd->devdata;
    
    switch (trip) {
    case 0: /* Warning */
        *temp = 75000;  /* 75°C */
        break;
    case 1: /* Critical */
        *temp = 85000;  /* 85°C */
        break;
    case 2: /* Emergency */
        *temp = 95000;  /* 95°C */
        break;
    default:
        return -EINVAL;
    }
    
    return 0;
}

static int milspec_thermal_notify(struct thermal_zone_device *tzd,
                                 int trip, enum thermal_trip_type type)
{
    pr_warn("MIL-SPEC: Thermal trip point %d reached\n", trip);
    
    if (trip >= 2) {
        /* Emergency thermal shutdown */
        pr_crit("MIL-SPEC: Emergency thermal shutdown!\n");
        if (milspec_state.mode5_level >= MODE5_PARANOID) {
            /* Paranoid mode - wipe before shutdown */
            milspec_emergency_wipe();
        } else {
            /* Normal thermal shutdown */
            orderly_poweroff(true);
        }
    }
    
    return 0;
}

static struct thermal_zone_device_ops milspec_thermal_ops = {
    .get_temp = milspec_thermal_get_temp,
    .get_trip_temp = milspec_thermal_get_trip_temp,
    .get_trip_type = milspec_thermal_get_trip_type,
    .notify = milspec_thermal_notify,
};
```

### **Phase 7: Latitude 5450 Hardware Initialization**

#### 7.1 Platform-Specific Initialization
```c
/* Initialize Latitude 5450 MIL-SPEC features */
static int milspec_latitude_5450_init(void)
{
    struct pci_dev *pdev;
    void __iomem *pch_base;
    int ret;
    
    pr_info("MIL-SPEC: Initializing Dell Latitude 5450 features\n");
    
    /* Map PCH registers */
    pch_base = ioremap(MTL_P_PCH_BASE, 0x10000);
    if (!pch_base) {
        pr_err("MIL-SPEC: Failed to map PCH registers\n");
        return -ENOMEM;
    }
    
    /* Check if MIL-SPEC features are present */
    u32 sku = ioread32(pch_base + 0x8000);
    if (!(sku & BIT(31))) {
        pr_warn("MIL-SPEC: SKU does not support MIL-SPEC features\n");
        iounmap(pch_base);
        return -ENODEV;
    }
    
    /* Initialize Intel ISH for sensor integration */
    pdev = pci_get_device(0x8086, 0x7e45, NULL);
    if (pdev) {
        pr_info("MIL-SPEC: Intel ISH found for sensor integration\n");
        milspec_caps.ish_present = true;
        pci_dev_put(pdev);
    }
    
    /* Configure I2C for ATECC608B */
    ret = milspec_init_i2c_controllers();
    if (ret)
        pr_warn("MIL-SPEC: I2C init failed: %d\n", ret);
    
    /* Configure GPIO controllers */
    ret = milspec_init_gpio_controllers();
    if (ret)
        pr_warn("MIL-SPEC: GPIO init failed: %d\n", ret);
    
    /* Enable Meteor Lake security features */
    ret = milspec_enable_mtl_security();
    if (ret)
        pr_warn("MIL-SPEC: MTL security init failed: %d\n", ret);
    
    iounmap(pch_base);
    return 0;
}

/* Enable Meteor Lake-P security features */
static int milspec_enable_mtl_security(void)
{
    u64 msr_val;
    
    /* Enable TME (Total Memory Encryption) if available */
    if (boot_cpu_has(X86_FEATURE_TME)) {
        rdmsrl(MSR_IA32_TME_ACTIVATE, msr_val);
        if (!(msr_val & TME_ACTIVATE_ENABLED)) {
            pr_info("MIL-SPEC: TME not enabled by BIOS\n");
        } else {
            pr_info("MIL-SPEC: TME enabled with %llu keys\n",
                    (msr_val >> 32) & 0xF);
        }
    }
    
    /* Configure CET (Control-flow Enforcement Technology) */
    if (boot_cpu_has(X86_FEATURE_CET)) {
        /* Enable CET for kernel protection */
        pr_info("MIL-SPEC: CET available for control flow protection\n");
    }
    
    return 0;
}
```

## 📊 **Implementation Priority**

### **High Priority:**
1. Latitude 5450 hardware initialization
2. Dell WMI integration with existing infrastructure  
3. CSME-based firmware update mechanism
4. Thread Director configuration for security workloads

### **Medium Priority:**
5. SMBIOS token implementation for MIL-SPEC range
6. ACPI device enumeration and method wrappers
7. Meteor Lake power state management
8. I2C controller setup for crypto chip

### **Low Priority:**
9. Advanced thermal management
10. Runtime ACPI table modifications
11. ISH sensor integration
12. NPU acceleration for crypto operations

## ⚠️ **Security Considerations**

1. **Firmware Authentication**: All firmware must be digitally signed
2. **SMBIOS Token Protection**: Security tokens require authentication
3. **ACPI Method Validation**: Validate all ACPI method parameters
4. **Power State Security**: Maintain security during suspend/resume
5. **Thermal Protection**: Prevent thermal-based attacks

## 📅 **Implementation Timeline**

- **Week 1**: ACPI device enumeration and method wrappers
- **Week 2**: SMBIOS token management and discovery
- **Week 3**: Firmware update infrastructure
- **Week 4**: Power management and thermal integration
- **Week 5**: Testing and security hardening

## 🔧 **Testing Strategy**

1. **ACPI Testing**: Verify all methods with acpidbg
2. **SMBIOS Testing**: Validate token operations
3. **Firmware Testing**: Test update/rollback scenarios
4. **Power Testing**: Suspend/resume cycling
5. **Thermal Testing**: Temperature threshold validation

## 📚 **Required Documentation**

1. ACPI specification compliance
2. Dell SMBIOS token documentation
3. Firmware update protocol specification
4. Power state transition diagrams
5. Thermal management policies

## 🔧 **Hardware-Specific Components**

### **Dell Latitude 5450 with Meteor Lake-P**
- Intel Core Ultra processor with hybrid architecture
- Thread Director for E-core/P-core optimization
- Intel CSME for secure firmware operations
- Intel ISH for sensor integration
- Multiple I2C controllers for crypto chip
- Extensive Dell WMI infrastructure
- TPM 2.0 with Intel TXT support

### **Key Integration Points**
1. **CSME HECI** (0000:00:16.0) - Secure firmware channel
2. **ISH** (0000:00:12.0) - Sensor hub for environmental monitoring
3. **I2C Controllers** (0000:00:15.x) - ATECC608B interface
4. **GPIO Controllers** (INTC1062) - Physical security signals
5. **Dell WMI GUIDs** - Event notifications and control

---

**Status**: Plan Ready for Implementation
**Priority**: High - Hardware-specific features critical
**Estimated Effort**: 5 weeks development
**Dependencies**: Intel Meteor Lake documentation, Dell WMI specs
**Hardware**: Dell Latitude 5450 with MIL-SPEC SKU