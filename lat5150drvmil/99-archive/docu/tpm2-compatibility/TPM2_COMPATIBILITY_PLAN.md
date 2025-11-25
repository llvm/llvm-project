# TPM2 COMPATIBILITY LAYER DEVELOPMENT PLAN

**PROJECT**: ME-TPM Compatibility Interface for Standard TPM2 Tools
**DATE**: 2025-09-22
**STATUS**: Planning Phase
**PRIORITY**: CRITICAL

---

## EXECUTIVE SUMMARY

**OBJECTIVE**: Develop a transparent compatibility layer that allows standard TPM2 tools to work with non-standard ME-coordinated TPM implementation without modification.

**CORE CHALLENGE**: TPM uses hex PCR addressing (0xCAFE, 0xBEEF) and ME command wrapping, making standard tpm2-tools incompatible.

**SOLUTION APPROACH**: Multi-layered compatibility interface with transparent translation between standard and ME-wrapped protocols.

---

## CURRENT STATE ANALYSIS

### EXISTING IMPLEMENTATION
- **ME-TPM Driver**: `/LAT5150_DEV/src/me_tpm_driver.py`
- **ME Bypass Code**: `/LAT5150_DEV/src/intel_me_tpm_bypass.c`
- **Enhanced Framework**: `/LAT5150_DEV/src/enhanced_tpm_framework.py`

### IDENTIFIED ISSUES
1. **Hex PCR Addressing**: 0x0000-0xFFFF range vs standard 0-23
2. **ME Command Wrapping**: All commands routed through Intel ME interface
3. **Military Token Dependency**: Requires tokens 0x049e-0x04a3 validation
4. **Non-standard Protocol**: Custom command structure vs TPM2 spec

### COMPATIBILITY REQUIREMENTS
- Support standard tpm2-tools without modification
- Transparent hex PCR translation
- ME command wrapping/unwrapping
- Military token integration
- Error handling and diagnostics

---

## ARCHITECTURE DESIGN

### LAYER 1: TRANSLATION INTERFACE
```
┌─────────────────────┐
│   Standard Apps     │ (tpm2_pcrread, tpm2_extend, etc.)
├─────────────────────┤
│ TPM2 Compatibility  │ ← OUR SOLUTION
│     Layer           │
├─────────────────────┤
│   ME-TPM Driver     │ (existing implementation)
├─────────────────────┤
│ Intel ME Interface  │ (/dev/mei0)
└─────────────────────┘
```

### LAYER 2: COMPONENT BREAKDOWN

#### A. PCR ADDRESS TRANSLATOR
```python
class PCRTranslator:
    def decimal_to_hex(self, pcr_decimal):
        """Convert 0-23 to 0x0000-0xFFFF mapping"""

    def hex_to_decimal(self, pcr_hex):
        """Convert hex PCR back to decimal for responses"""

    def validate_pcr_range(self, pcr):
        """Validate PCR is in supported range"""
```

#### B. COMMAND WRAPPER
```python
class MECommandWrapper:
    def wrap_tpm2_command(self, standard_cmd):
        """Wrap standard TPM2 command in ME protocol"""

    def unwrap_me_response(self, me_response):
        """Extract TPM2 response from ME wrapper"""

    def build_me_header(self, cmd_type, payload_size):
        """Build ME protocol header (magic: 0xCAFEBABE)"""
```

#### C. PROTOCOL BRIDGE
```python
class TPM2Bridge:
    def intercept_tpm_calls(self):
        """Intercept standard TPM device calls"""

    def route_to_me_interface(self, cmd):
        """Route commands to ME-TPM driver"""

    def emulate_standard_responses(self, me_data):
        """Format responses as standard TPM2"""
```

---

## IMPLEMENTATION PHASES

### PHASE 1: CORE TRANSLATION (Week 1-2)
**Deliverables**:
- PCR address translator (decimal ↔ hex)
- Basic ME command wrapper
- Command structure documentation

**Key Components**:
```bash
# Files to create:
tpm2_compat/
├── pcr_translator.py     # PCR addressing translation
├── me_wrapper.py         # ME command wrapping
├── protocol_bridge.py    # TPM2 ↔ ME protocol bridge
└── constants.py          # Magic values, mappings
```

### PHASE 2: DEVICE EMULATION (Week 3)
**Deliverables**:
- `/dev/tpm0` device emulation
- Standard TPM2 API compatibility
- Error handling and logging

**Key Components**:
```bash
# Files to create:
tpm2_compat/
├── device_emulator.py    # /dev/tpm0 emulation
├── api_compatibility.py  # Standard TPM2 API layer
└── error_handler.py      # Error translation
```

### PHASE 3: TOOL INTEGRATION (Week 4)
**Deliverables**:
- tpm2-tools compatibility testing
- Performance optimization
- Integration documentation

**Key Components**:
```bash
# Files to create:
tools/
├── tpm2_test_suite.py    # Compatibility tests
├── performance_monitor.py # Performance analysis
└── integration_guide.md   # Setup documentation
```

---

## TECHNICAL SPECIFICATIONS

### COMMAND STRUCTURE MAPPING

#### Standard TPM2 Command:
```c
struct tpm2_command {
    uint16_t tag;           // TPM_ST_SESSIONS
    uint32_t length;        // Command length
    uint32_t command_code;  // TPM_CC_*
    uint8_t  payload[];     // Command data
};
```

#### ME-Wrapped Command:
```c
struct me_tpm_command {
    uint32_t magic;         // 0xCAFEBABE
    uint16_t me_header_len; // ME header length
    uint8_t  me_session_id[16]; // Session identifier
    uint16_t pcr_index;     // Hex PCR (0x0000-0xFFFF)
    uint8_t  operation;     // Read/Write/Extend/Reset
    uint8_t  hash_alg;      // SHA256/SHA512/SM3
    uint32_t tpm2_cmd_len;  // Wrapped TPM2 command length
    uint8_t  tpm2_cmd[];    // Original TPM2 command
    uint32_t checksum;      // ME protocol checksum
};
```

### PCR MAPPING STRATEGY

#### Standard → Hex Translation:
```python
PCR_MAP = {
    # Standard PCRs (0-7: BIOS/UEFI)
    0: 0x0000,   # BIOS measurements
    1: 0x0001,   # BIOS configuration
    2: 0x0002,   # Option ROM Code
    3: 0x0003,   # Option ROM Config
    4: 0x0004,   # IPL (Master Boot Record)
    5: 0x0005,   # IPL Config
    6: 0x0006,   # State Transition/Wake
    7: 0x0007,   # Platform Manufacturer

    # OS PCRs (8-15)
    8: 0x0008,   # OS Loader
    9: 0x0009,   # OS Configuration
    10: 0x000A,  # IMA Template
    11: 0x000B,  # Kernel Command Line
    12: 0x000C,  # Kernel Modules
    13: 0x000D,  # OS Boot
    14: 0x000E,  # MokList
    15: 0x000F,  # System Boot

    # Extended PCRs (16-23)
    16: 0x0010,  # Debug
    17: 0x0011,  # Dynamic Root of Trust
    18: 0x0012,  # Trusted OS
    19: 0x0013,  # Trusted OS Config
    20: 0x0014,  # Trusted OS Data
    21: 0x0015,  # OS Applications
    22: 0x0016,  # OS Application Config
    23: 0x0017,  # OS Application Data

    # Special Configuration PCRs
    'CAFE': 0xCAFE,  # Algorithm configuration
    'BEEF': 0xBEEF,  # Extended functionality
    'DEAD': 0xDEAD,  # Debug/diagnostic
    'FACE': 0xFACE,  # Factory configuration
}
```

---

## COORDINATION REQUIREMENTS

### HARDWARE-INTEL COORDINATION
**Required Information**:
- ME interface command structure documentation
- ME session management protocols
- Hardware register mappings for TPM-ME coordination
- Error codes and status interpretations

**Deliverables Needed**:
- ME protocol specification document
- Command format examples
- Error handling guidelines

### HARDWARE-DELL COORDINATION
**Required Information**:
- Military token integration requirements
- Dell-specific TPM implementation details
- SMBIOS token validation procedures
- Platform-specific ME configurations

**Deliverables Needed**:
- Military token specification
- Platform configuration guide
- Validation procedures

### C-INTERNAL COORDINATION
**Required Development**:
- Low-level ME interface code
- Performance-critical translation functions
- Device driver integration
- System-level compatibility layer

**Deliverables Needed**:
- C library for ME communication
- Kernel module for device emulation
- Performance-optimized translators

---

## TESTING STRATEGY

### UNIT TESTING
```bash
# Test individual components
python -m pytest tests/test_pcr_translator.py
python -m pytest tests/test_me_wrapper.py
python -m pytest tests/test_protocol_bridge.py
```

### INTEGRATION TESTING
```bash
# Test with real tpm2-tools
tpm2_pcrread sha256:0,1,2,3  # Should work transparently
tpm2_extend 7:sha256=abcd...  # Should translate to hex PCR
tpm2_quote -c key.ctx -l sha256:0,1,2  # Full attestation test
```

### COMPATIBILITY TESTING
```bash
# Standard tools that must work
tpm2_startup -c
tpm2_pcrread
tpm2_extend
tpm2_quote
tpm2_createprimary
tpm2_create
tpm2_load
```

---

## RISK MITIGATION

### TECHNICAL RISKS
1. **Performance Impact**: Translation overhead slowing operations
   - *Mitigation*: C-based critical path implementation

2. **Protocol Compatibility**: ME protocol changes breaking compatibility
   - *Mitigation*: Version detection and adaptive protocols

3. **Error Propagation**: ME errors not translating correctly to standard errors
   - *Mitigation*: Comprehensive error mapping tables

### OPERATIONAL RISKS
1. **Tool Detection**: Security tools detecting non-standard behavior
   - *Mitigation*: Perfect emulation of standard TPM responses

2. **Military Token Dependency**: Operations failing without tokens
   - *Mitigation*: Graceful degradation to limited functionality

---

## SUCCESS CRITERIA

### PRIMARY OBJECTIVES
- ✅ Standard tpm2-tools work without modification
- ✅ Hex PCR addressing transparent to applications
- ✅ Military token integration seamless
- ✅ Performance within 10% of direct TPM access

### SECONDARY OBJECTIVES
- ✅ Enterprise TPM management tools compatible
- ✅ Attestation and quote operations functional
- ✅ Advanced algorithms (52+) accessible
- ✅ Configuration PCRs (0xCAFE, 0xBEEF) operational

---

## TIMELINE

**WEEK 1**: Core translation components (PCR, commands)
**WEEK 2**: ME protocol wrapper and bridge
**WEEK 3**: Device emulation and API compatibility
**WEEK 4**: Integration testing and documentation

**TOTAL ESTIMATED EFFORT**: 40-80 hours as per technical brief
**DELIVERY TARGET**: End of Week 4

---

## CONCLUSION

This compatibility layer will provide seamless integration between standard TPM2 tools and the non-standard ME-coordinated TPM implementation. Success will enable full operational capability while maintaining security through military token integration.

**NEXT ACTIONS**: Coordinate with HARDWARE-INTEL, HARDWARE-DELL, and C-INTERNAL teams for detailed specifications and implementation support.