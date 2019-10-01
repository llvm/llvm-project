// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430c111 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-C111 %s

// MSP430-C111: clang{{.*}} "-cc1" {{.*}} "-D__MSP430C111__"
// MSP430-C111: msp430-elf-ld{{.*}} "-Tmsp430c111.ld"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=msp430i2020 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-I2020 %s

// MSP430-I2020: clang{{.*}} "-cc1" {{.*}} "-D__MSP430i2020__"
// MSP430-I2020: msp430-elf-ld{{.*}} "-Tmsp430i2020.ld"

// RUN: %clang %s -### -no-canonical-prefixes -target msp430 -mmcu=not-a-mcu 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430-UNSUP %s

// MSP430-UNSUP: error: the clang compiler does not support 'not-a-mcu'
