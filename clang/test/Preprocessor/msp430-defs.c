// RUN: %clang_cc1 -triple msp430 -E -dM %s -o - \
// RUN:   | FileCheck -check-prefix=MSP430 %s
// RUN: %clang_cc1 -triple msp430 -target-cpu msp430 -E -dM %s -o - \
// RUN:   | FileCheck -check-prefix=MSP430 %s
// MSP430-NOT: #define __MSP430X__

// RUN: %clang_cc1 -triple msp430 -target-cpu msp430x -E -dM %s -o - \
// RUN:   | FileCheck -check-prefix=MSP430X %s
// MSP430X: #define __MSP430X__ 1
