// RUN: %clang_cc1 -triple armv7s-linux-gnu -emit-llvm -o - %s \
// RUN:     -target-feature +neon -target-cpu cortex-a8 \
// RUN:     -fsanitize=signed-integer-overflow \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=ARMV7

// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm -o - %s \
// RUN:     -target-feature +neon -target-cpu cortex-a53 \
// RUN:     -fsanitize=signed-integer-overflow \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=AARCH64

// Verify we emit constants for "immediate" builtin arguments.
// Emitting a scalar expression can make the immediate be generated as
// overflow intrinsics, if the overflow sanitizer is enabled.

// PR23517

#include <arm_neon.h>

int32x2_t test_vqrshrn_n_s64(int64x2_t a) {
  // CHECK-LABEL: @test_vqrshrn_n_s64
  // CHECK-AARCH64: call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> {{.*}}, i32 1)
  // CHECK-ARMV7: call <2 x i32> @llvm.arm.neon.vqrshiftns.v2i32(<2 x i64> {{.*}}, <2 x i64> <i64 -1, i64 -1>)
  return vqrshrn_n_s64(a, 0 + 1);
}
