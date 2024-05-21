// RUN: %clang_cc1 -triple arm64-apple-ios -target-feature +neon -ffreestanding -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-UNSIGNED-POLY %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -ffreestanding -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-UNSIGNED-POLY %s
// RUN: %clang_cc1 -triple armv7-apple-ios -ffreestanding -target-cpu cortex-a8 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-SIGNED-POLY %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// Polynomial types really should be universally unsigned, otherwise casting
// (say) poly8_t "x^7" to poly16_t would change it to "x^15 + x^14 + ... +
// x^7". Unfortunately 32-bit ARM ended up in a slightly delicate ABI situation
// so for now it got that wrong.

poly16_t test_poly8(poly8_t pIn) {
// CHECK-UNSIGNED-POLY: @_Z10test_poly8h
// CHECK-UNSIGNED-POLY: zext i8 {{.*}} to i16

// CHECK-SIGNED-POLY: @_Z10test_poly8a
// CHECK-SIGNED-POLY: sext i8 {{.*}} to i16

  return pIn;
}
