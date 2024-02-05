// RUN: %clang_cc1 -triple arm64-apple-ios -target-feature +neon -ffreestanding -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -target-feature +neon -ffreestanding -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-ios -ffreestanding -target-cpu cortex-a8 -S -emit-llvm -o - %s | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

poly16_t test_poly8(poly8_t pIn) {
// CHECK: @_Z10test_poly8h
// CHECK: zext i8 {{.*}} to i16

  return pIn;
}
