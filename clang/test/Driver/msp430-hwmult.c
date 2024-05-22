// Test that different values of -mhwmult pick correct
// MSP430 hwmult target-feature(s).

// RUN: %clang -### -target msp430 %s 2>&1 | FileCheck %s
// RUN: %clang -### -target msp430 %s -mhwmult=auto 2>&1 | FileCheck %s
// CHECK-NOT: "-target-feature" "+hwmult16"
// CHECK-NOT: "-target-feature" "+hwmult32"
// CHECK-NOT: "-target-feature" "+hwmultf5"

// RUN: %clang -### -target msp430 %s -mhwmult=none 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target msp430 %s -mhwmult=none -mmcu=msp430f147 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// RUN: %clang -### -target msp430 %s -mhwmult=none -mmcu=msp430f4783 2>&1 | FileCheck --check-prefix=CHECK-NONE %s
// CHECK-NONE: "-target-feature" "-hwmult16"
// CHECK-NONE: "-target-feature" "-hwmult32"
// CHECK-NONE: "-target-feature" "-hwmultf5"

// RUN: %clang -### -target msp430 %s -mhwmult=16bit 2>&1 | FileCheck --check-prefix=CHECK-16 %s
// CHECK-16: "-target-feature" "+hwmult16"

// RUN: %clang  -### -target msp430 %s -mhwmult=32bit 2>&1 | FileCheck --check-prefix=CHECK-32 %s
// CHECK-32: "-target-feature" "+hwmult32"

// RUN: %clang  -### -target msp430 %s -mhwmult=f5series 2>&1 | FileCheck --check-prefix=CHECK-F5 %s
// CHECK-F5: "-target-feature" "+hwmultf5"

// RUN: not %clang  -### --target=msp430 %s -mhwmult=rrr 2>&1 | FileCheck --check-prefix=INVL-ARG %s
// INVL-ARG: error: unsupported argument 'rrr' to option '-mhwmult='

// RUN: %clang  -### -target msp430 %s -mhwmult=auto 2>&1 | FileCheck --check-prefix=WRN-NODEV %s
// WRN-NODEV: warning: no MCU device specified, but '-mhwmult' is set to 'auto',
//            assuming no hardware multiply; use '-mmcu' to specify a MSP430 device,
//            or '-mhwmult' to set hardware multiply type explicitly.

// RUN: %clang  -### -target msp430 %s -mhwmult=16bit -mmcu=msp430c111 2>&1 | FileCheck --check-prefix=WRN-UNSUP %s
// RUN: %clang  -### -target msp430 %s -mhwmult=32bit -mmcu=msp430c111 2>&1 | FileCheck --check-prefix=WRN-UNSUP %s
// RUN: %clang  -### -target msp430 %s -mhwmult=f5series -mmcu=msp430c111 2>&1 | FileCheck --check-prefix=WRN-UNSUP %s
// WRN-UNSUP: warning: the given MCU does not support hardware multiply, but '-mhwmult' is set to

// RUN: %clang  -### -target msp430 %s -mhwmult=16bit -mmcu=msp430f4783 2>&1 | FileCheck --check-prefix=WRN-MISMCH %s
// RUN: %clang  -### -target msp430 %s -mhwmult=32bit -mmcu=msp430f147 2>&1 | FileCheck --check-prefix=WRN-MISMCH %s
// RUN: %clang  -### -target msp430 %s -mhwmult=f5series -mmcu=msp430f4783 2>&1 | FileCheck --check-prefix=WRN-MISMCH %s
// WRN-MISMCH: warning: the given MCU supports {{.*}} hardware multiply, but '-mhwmult' is set to {{.*}}
