# ATECC608B Crypto Chip Support

## Overview

The Dell MIL-SPEC driver includes **optional** support for the ATECC608B secure element. This hardware crypto chip provides:

- Hardware-based key storage
- Secure key generation
- ECDSA signing and verification
- Hardware random number generation
- Tamper-resistant secure storage

## Optional Hardware

**IMPORTANT**: The ATECC608B is OPTIONAL hardware that may not be installed on all systems. The driver will:

1. Automatically detect if the chip is present during initialization
2. Enable hardware crypto features if found
3. Continue normal operation using software crypto if not found

## Detection Process

The driver searches for the ATECC608B on common I2C buses:
- Bus 3 and 7 (typical Dell locations)
- Bus 1 and 0 (fallback locations)
- I2C address: 0x60

## Status Reporting

You can check if the crypto chip is detected:

```bash
# Via sysfs
cat /sys/devices/platform/dell-milspec/crypto_status

# Via test program
./test-milspec -s

# Expected output when NOT installed:
# ATECC608B: not installed
# Status: Using software crypto
# Note: Hardware crypto is optional
```

## Driver Behavior

### When Crypto Chip Present:
- Hardware crypto acceleration enabled
- MMIO crypto status register set
- Boot progress includes CRYPTO stage
- Emergency wipe includes crypto chip secure erase

### When Crypto Chip Absent:
- Software crypto used instead
- No error messages (this is normal)
- All other features work normally
- Boot continues without CRYPTO stage

## Installation (If Needed)

If you need to install an ATECC608B:

1. The chip must be connected to an I2C bus
2. Default address is 0x60
3. Requires pull-up resistors on SDA/SCL
4. Must be provisioned with keys before use

## Testing

To verify crypto detection:

```bash
# Load module with debug
sudo insmod dell-milspec.ko milspec_debug=0xFF milspec_force=1

# Check kernel log
dmesg | grep -i crypto

# Expected when not present:
# MIL-SPEC: Checking for optional ATECC608B crypto chip...
# MIL-SPEC: ATECC608B crypto chip not detected - continuing without hardware crypto
# MIL-SPEC: This is normal if the chip is not installed
```

## Development Notes

- The crypto chip is accessed via standard Linux I2C APIs
- Wake sequence requires 1.5ms delay
- All crypto operations check `crypto_chip.present` before use
- I2C adapters are properly reference counted (get/put)

## Security Considerations

- Hardware crypto provides better key protection than software
- Keys stored in ATECC608B cannot be extracted
- Emergency wipe will securely erase crypto chip if present
- Software crypto is still secure but keys are in system memory