# Intel ME Interface Specifications for TPM2 Compatibility Layer

**CLASSIFICATION**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**DATE**: 22 SEP 2025
**SOURCE**: HARDWARE-INTEL Agent Coordination
**PROJECT**: ME-TPM Compatibility Interface Development

---

## Executive Summary

Comprehensive Intel ME interface specifications for developing TPM2 compatibility layer with non-standard ME-coordinated TPM implementation. Enables standard tpm2-tools to work transparently with hex PCR addressing and ME command wrapping.

---

## 1. ME Command Structure Documentation

### ME Protocol Header Format
```c
typedef struct {
    uint8_t  command;        // Command identifier
    uint8_t  reserved;       // Reserved field (must be 0)
    uint16_t length;         // Message length (including header)
    uint32_t session_id;     // Session identifier
    uint8_t  payload[];      // Variable-length payload
} mei_message_header_t;
```

**Note**: The 0xCAFEBABE magic value mentioned in reference code is not part of standard Intel MEI protocol. Intel MEI uses GUID-based client identification instead of magic numbers.

### Session Management Structure
```c
typedef struct {
    uuid_t   client_guid;    // Client GUID for TPM coordination
    uint32_t session_id;     // Unique session identifier
    uint8_t  security_level; // Security classification level
    uint32_t capabilities;   // Available ME capabilities
} mei_tpm_session_t;

// TPM-specific ME client GUID (example - vendor specific)
#define ME_TPM_CLIENT_GUID { 0x12345678, 0x1234, 0x5678, \
                            { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 } }
```

---

## 2. Hardware Register Mappings

### ME Base Address Configuration
```c
#define ME_BASE_ADDR        0xFED1A000  // Intel ME MMIO base address
#define ME_REGION_SIZE      0x1000      // 4KB register space

// Core ME Registers
#define ME_H_CSR            0x04        // Host Control Status Register
#define ME_ME_CSR_HA        0x0C        // ME Control Status Register
#define ME_H_GS             0x4C        // Host General Status Register

// TPM-ME Coordination Registers (Dell-specific extensions)
#define ME_TPM_STATE        0x40        // TPM coordination state
#define ME_TPM_CTRL         0x44        // TPM coordination control

// HAP Mode Control
#define HAP_CTRL_REG        0x50        // HAP mode control register
#define HAP_DISABLE_BIT     (1 << 0)    // HAP disable bit
#define TPM_BYPASS_HAP      (1 << 16)   // TPM bypass in HAP mode
```

### Status and Control Register Bit Definitions
```c
// ME_H_CSR bit definitions
#define ME_H_CSR_IE         (1 << 0)    // Interrupt Enable
#define ME_H_CSR_IS         (1 << 1)    // Interrupt Status
#define ME_H_CSR_IG         (1 << 2)    // Interrupt Generate
#define ME_H_CSR_RDY        (1 << 3)    // Ready
#define ME_H_CSR_RST        (1 << 4)    // Reset

// ME_TPM_STATE bit definitions
#define TPM_STATE_READY     (1 << 0)    // TPM ready for commands
#define TPM_STATE_ME_COORD  (1 << 1)    // ME coordination active
#define TPM_STATE_BYPASS    (1 << 31)   // TPM bypass mode
```

---

## 3. Interface Specifications

### /dev/mei0 Device Communication
```c
// MEI IOCTLs for TPM coordination
#define IOCTL_MEI_CONNECT_CLIENT    _IOWR('H', 0x01, struct mei_connect_client_data)
#define IOCTL_MEI_NOTIFY_SET        _IOW('H', 0x02, uint32_t)
#define IOCTL_MEI_NOTIFY_GET        _IOR('H', 0x03, uint32_t)

// Connection structure for TPM client
struct mei_connect_client_data {
    uuid_t in_client_uuid;      // TPM client GUID
    uint8_t out_client_uuid[16]; // Connected client UUID
    uint32_t out_mtu;           // Maximum transmission unit
    uint8_t out_version;        // Protocol version
    uint8_t out_reserved[3];    // Reserved
};
```

### Buffer Sizes and Timing Constraints
```c
#define ME_TPM_MAX_MESSAGE_SIZE     512     // Maximum message size
#define ME_TPM_TIMEOUT_MS          5000     // Command timeout (5 seconds)
#define ME_TPM_RETRY_COUNT           3      // Maximum retry attempts
#define ME_CONNECTION_TIMEOUT_MS   10000    // Connection timeout (10 seconds)
```

---

## 4. Protocol Translation Requirements

### TPM2 Command Wrapping for ME Interface
```c
typedef struct {
    mei_message_header_t header;    // ME message header
    uint16_t tpm_command_tag;       // TPM command tag
    uint32_t tpm_command_size;      // TPM command size
    uint32_t tpm_command_code;      // TPM command code
    uint8_t  tpm_command_data[];    // TPM command payload
} me_wrapped_tpm_command_t;

// ME-TPM command wrapping function
int wrap_tpm_command_for_me(const uint8_t *tpm_cmd, size_t tpm_cmd_len,
                           me_wrapped_tpm_command_t **wrapped_cmd) {
    size_t wrapped_size = sizeof(me_wrapped_tpm_command_t) + tmp_cmd_len;
    *wrapped_cmd = malloc(wrapped_size);

    // Fill ME header
    (*wrapped_cmd)->header.command = ME_TPM_COMMAND;
    (*wrapped_cmd)->header.reserved = 0;
    (*wrapped_cmd)->header.length = wrapped_size;
    (*wrapped_cmd)->header.session_id = get_current_session_id();

    // Copy TPM command
    memcpy(&(*wrapped_cmd)->tpm_command_tag, tpm_cmd, tpm_cmd_len);

    return 0;
}
```

### Response Unwrapping Procedures
```c
typedef struct {
    mei_message_header_t header;    // ME response header
    uint8_t  status;               // ME operation status
    uint8_t  reserved[3];          // Reserved
    uint16_t tpm_response_tag;     // TPM response tag
    uint32_t tpm_response_size;    // TPM response size
    uint32_t tpm_response_code;    // TPM response code
    uint8_t  tpm_response_data[];  // TPM response payload
} me_wrapped_tpm_response_t;

// Response unwrapping function
int unwrap_tpm_response_from_me(const me_wrapped_tpm_response_t *wrapped_resp,
                               uint8_t **tpm_resp, size_t *tpm_resp_len) {
    if (wrapped_resp->status != ME_STATUS_SUCCESS) {
        return -wrapped_resp->status;
    }

    *tpm_resp_len = wrapped_resp->tpm_response_size;
    *tpm_resp = malloc(*tpm_resp_len);
    memcpy(*tpm_resp, &wrapped_resp->tpm_response_tag, *tpm_resp_len);

    return 0;
}
```

---

## 5. Error Codes and Status Interpretations

```c
// ME-TPM coordination error codes
#define ME_STATUS_SUCCESS           0x00    // Operation successful
#define ME_STATUS_INVALID_STATE     0x01    // Invalid ME state
#define ME_STATUS_NOT_READY         0x02    // TPM not ready
#define ME_STATUS_TIMEOUT           0x03    // Operation timeout
#define ME_STATUS_INVALID_COMMAND   0x04    // Invalid command
#define ME_STATUS_INSUFFICIENT_AUTH 0x05    // Insufficient authorization
#define ME_STATUS_HAP_RESTRICTION   0x06    // HAP mode restriction
#define ME_STATUS_TPM_ERROR         0x80    // TPM-specific error (+ TPM error code)

// Error interpretation function
const char* interpret_me_tpm_error(uint8_t error_code) {
    switch (error_code) {
        case ME_STATUS_SUCCESS:           return "Success";
        case ME_STATUS_INVALID_STATE:     return "Invalid ME state";
        case ME_STATUS_NOT_READY:         return "TPM not ready";
        case ME_STATUS_TIMEOUT:           return "Operation timeout";
        case ME_STATUS_INVALID_COMMAND:   return "Invalid command";
        case ME_STATUS_INSUFFICIENT_AUTH: return "Insufficient authorization";
        case ME_STATUS_HAP_RESTRICTION:   return "HAP mode restriction";
        default:
            if (error_code >= ME_STATUS_TPM_ERROR) {
                return "TPM-specific error";
            }
            return "Unknown error";
    }
}
```

---

## 6. PCR Addressing Translation

### Extended PCR Addressing for Hex Range
```c
// PCR addressing translation for extended range
typedef struct {
    uint16_t extended_pcr_index;    // Extended PCR index (0x0000-0xFFFF)
    uint8_t  standard_pcr_index;    // Standard PCR index (0-23)
    uint8_t  bank_selector;         // Algorithm bank selector
    uint32_t algorithm_id;          // Crypto algorithm identifier
} pcr_address_translation_t;

// Translation function for extended PCR addressing
int translate_extended_pcr_address(uint16_t extended_pcr,
                                  pcr_address_translation_t *translation) {
    // Map extended range to standard PCRs and algorithm banks
    translation->standard_pcr_index = extended_pcr & 0x1F;  // PCR 0-31
    translation->bank_selector = (extended_pcr >> 5) & 0x7F; // Bank selector

    // Map to specific algorithms based on bank selector
    switch (translation->bank_selector) {
        case 0: translation->algorithm_id = TPM2_ALG_SHA256; break;
        case 1: translation->algorithm_id = TPM2_ALG_SHA384; break;
        case 2: translation->algorithm_id = TPM2_ALG_SHA3_256; break;
        case 3: translation->algorithm_id = TPM2_ALG_SHA3_384; break;
        default: return -EINVAL;
    }

    return 0;
}
```

---

## 7. Security Considerations

### ME Firmware Version Compatibility
```c
// ME firmware version compatibility check
int check_me_firmware_compatibility(void) {
    char fw_version[32];
    FILE *fp = fopen("/sys/class/mei/mei0/fw_ver", "r");
    if (!fp) return -ENODEV;

    if (fgets(fw_version, sizeof(fw_version), fp)) {
        // Check for compatible firmware versions
        // Version format: major.minor.hotfix.build
        fclose(fp);
        return verify_fw_version_compatibility(fw_version);
    }

    fclose(fp);
    return -EINVAL;
}

// HAP mode security validation
int validate_hap_mode_security(void) {
    uint32_t me_status = read_me_register(ME_H_GS);

    // Verify HAP mode is active (0x94000245 as mentioned in existing code)
    if ((me_status & 0xFFFF0000) != 0x94000000) {
        return -EPERM;  // Not in HAP mode
    }

    // Additional security validations
    return validate_me_security_posture();
}
```

---

## 8. Integration Recommendations

1. **Transparent Compatibility Layer**: Implement shim library that intercepts tpm2-tools calls and translates them through ME interface.

2. **Session Management**: Maintain persistent ME sessions for better performance and reduced connection overhead.

3. **Error Handling**: Implement robust error handling that can distinguish between ME errors and TPM errors.

4. **Performance Optimization**: Use asynchronous I/O for ME communication to avoid blocking standard TPM operations.

5. **Security Validation**: Always validate ME firmware version and HAP mode status before allowing TPM operations.

---

**PREPARED BY**: HARDWARE-INTEL Agent
**REVIEWED BY**: Technical Operations
**STATUS**: Specification Complete
**CLASSIFICATION**: UNCLASSIFIED // FOR OFFICIAL USE ONLY