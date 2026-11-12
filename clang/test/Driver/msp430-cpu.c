// RUN: %clang -### -c %s --target=msp430 2>&1 \
// RUN:   | FileCheck -check-prefix=NO-CPU %s
// NO-CPU-NOT: "-target-cpu"

// RUN: %clang -### -c %s --target=msp430 -mcpu=msp430 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430 %s
// RUN: %clang -### -c %s --target=msp430 -mmcu=msp430c111 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430 %s
// MSP430: "-cc1"{{.*}} "-target-cpu" "msp430"

// msp430xv2 is accepted for GCC compatibility and treated as msp430x.
// RUN: %clang -### -c %s --target=msp430 -mcpu=msp430x 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430X %s
// RUN: %clang -### -c %s --target=msp430 -mcpu=msp430xv2 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430X %s
// RUN: %clang -### -c %s --target=msp430 -mmcu=msp430f2617 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430X %s
// RUN: %clang -### -c %s --target=msp430 -mmcu=msp430f5529 2>&1 \
// RUN:   | FileCheck -check-prefix=MSP430X %s
// MSP430X: "-cc1"{{.*}} "-target-cpu" "msp430x"

// RUN: not %clang -c %s --target=msp430 -mcpu=not-a-cpu -o /dev/null 2>&1 \
// RUN:   | FileCheck -check-prefix=INVALID %s
// INVALID: error: unknown target CPU 'not-a-cpu'
// INVALID-NEXT: note: valid target CPU values are: msp430, msp430x
