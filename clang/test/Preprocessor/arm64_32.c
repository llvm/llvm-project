// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64_32-apple-ios < /dev/null | FileCheck %s --check-prefix=CHECK-32
// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64-apple-ios < /dev/null | FileCheck %s --check-prefix=CHECK-64

// CHECK-32: #define __ARM64_ARCH_8_32__ 1
// CHECK-64: #define __ARM64_ARCH_8__ 1
