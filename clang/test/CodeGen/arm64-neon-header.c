// RUN: %clang_cc1 -triple arm64-pc-windows-msvc -target-feature +neon \
// RUN:     -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:     -emit-llvm -o - %s | FileCheck %s

#include <arm64_neon.h>

// CHECK-LABEL: define{{.*}} @test_vld1q_s32_x4(
// CHECK-NOT: neon_ld1m4_q32
// CHECK: call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld1x4.v4i32.p0(ptr {{.*}})
// CHECK-NOT: neon_ld1m4_q32
// CHECK: ret
int32x4x4_t test_vld1q_s32_x4(int32_t const *a) {
  return vld1q_s32_x4(a);
}
