// RUN: %clang_cc1 -E -dM -triple avr-unknown-unknown -target-cpu atmega328p /dev/null | FileCheck -match-full-lines %s

// CHECK: #define AVR 1
// CHECK: #define __AVR 1
// CHECK: #define __AVR_ARCH__ 5
// CHECK: #define __AVR_ATmega328P__ 1
// CHECK-NOT: #define __AVR_HAVE_EIJMP_EICALL__
// CHECK: #define __AVR_HAVE_LPMX__ 1
// CHECK: #define __AVR_HAVE_MOVW__ 1
// CHECK: #define __AVR_HAVE_MUL__ 1
// CHECK: #define __AVR__ 1
// CHECK: #define __ELF__ 1
