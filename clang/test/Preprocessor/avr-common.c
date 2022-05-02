// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown -target-cpu attiny13 /dev/null | FileCheck -match-full-lines --check-prefixes=CHECK,AVR %s
// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown -target-cpu attiny4 /dev/null | FileCheck -match-full-lines --check-prefixes=CHECK,TINY %s

// CHECK: #define AVR 1
// CHECK: #define __AVR 1

// TINY: #define __AVR_TINY__ 1
// AVR-NOT: __AVR_TINY__

// CHECK: #define __AVR__ 1
// CHECK: #define __ELF__ 1
