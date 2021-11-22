// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | opt -S -mem2reg | FileCheck %s
// Test ARM64 SIMD vcreate intrinsics

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

float32x2_t test_vcreate_f32(uint64_t a1) {
  // CHECK: test_vcreate_f32
  return vcreate_f32(a1);
  // CHECK: bitcast {{.*}} to <2 x float>
  // CHECK-NEXT: ret
}
