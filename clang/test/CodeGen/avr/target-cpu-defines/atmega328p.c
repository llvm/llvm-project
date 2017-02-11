// REQUIRES: avr-registered-target
// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown -target-cpu atmega328p /dev/null | FileCheck -match-full-lines %s

// CHECK: #define AVR 1
// CHECK: #define __AVR 1
// CHECK: #define __AVR_ATmega328P__ 1
// CHECK: #define __AVR__ 1
