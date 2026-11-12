// __BITINT_MAXWIDTH__ is defined only where the target has _BitInt.

// RUN: %clang_cc1 -E -dM -triple=x86_64-unknown-linux-gnu < /dev/null | FileCheck --check-prefix=BITINT %s
// RUN: %clang_cc1 -E -dM -triple=riscv32-unknown-elf < /dev/null | FileCheck --check-prefix=BITINT %s
// RUN: %clang_cc1 -E -dM -triple=avr < /dev/null | FileCheck --check-prefix=NOBITINT %s
// RUN: %clang_cc1 -E -dM -triple=msp430 < /dev/null | FileCheck --check-prefix=NOBITINT %s
// RUN: %clang_cc1 -E -dM -triple=m68k < /dev/null | FileCheck --check-prefix=NOBITINT %s

// BITINT: #define __BITINT_MAXWIDTH__ {{[0-9]+}}
// NOBITINT-NOT: __BITINT_MAXWIDTH__
