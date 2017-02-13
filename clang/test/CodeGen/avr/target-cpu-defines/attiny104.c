// REQUIRES: avr-registered-target
// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown -target-cpu attiny104 /dev/null | FileCheck -match-full-lines %s

// CHECK: #define AVR 1
// CHECK: #define __AVR 1
// CHECK: #define __AVR_ATtiny104__ 1
// CHECK: #define __AVR__ 1
