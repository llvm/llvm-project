// RUN: %clang_cc1 -E -dM -triple thumbv7m-apple-unknown-macho -target-cpu cortex-m3 %s | FileCheck %s -check-prefix CHECK-7M

// CHECK-7M: #define __APPLE_CC__
// CHECK-7M: #define __APPLE__
// CHECK-7M: #define __ARM_ARCH_7M__
// CHECK-7M-NOT: #define __MACH__

// RUN: %clang_cc1 -E -dM -triple thumbv7em-apple-unknown-macho -target-cpu cortex-m4 %s | FileCheck %s -check-prefix CHECK-7EM

// CHECK-7EM: #define __APPLE_CC__
// CHECK-7EM: #define __APPLE__
// CHECK-7EM: #define __ARM_ARCH_7EM__
// CHECK-7EM-NOT: #define __MACH__

// RUN: %clang_cc1 -E -dM -triple thumbv6m-apple-unknown-macho -target-cpu cortex-m0 %s | FileCheck %s -check-prefix CHECK-6M

// CHECK-6M: #define __APPLE_CC__
// CHECK-6M: #define __APPLE__
// CHECK-6M: #define __ARM_ARCH_6M__
// CHECK-6M-NOT: #define __MACH__
