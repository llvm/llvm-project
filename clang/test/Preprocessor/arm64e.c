// RUN: %clang_cc1 -E -dM -ffreestanding -triple=arm64e-apple-ios < /dev/null | FileCheck %s

// CHECK: #define __ARM64_ARCH_8__ 1
// CHECK: #define __arm64__ 1
// CHECK: #define __arm64e__ 1
