// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown /dev/null | FileCheck -match-full-lines %s

// CHECK: #define AVR 1
// CHECK: #define __AVR 1
// CHECK: #define __AVR__ 1
// CHECK: #define __ELF__ 1
