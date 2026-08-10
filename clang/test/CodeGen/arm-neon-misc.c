// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumbv7-apple-darwin \
// RUN:   -target-abi apcs-gnu \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -mfloat-abi soft \
// RUN:   -target-feature +soft-float-abi \
// RUN:   -ffreestanding \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

#include <arm_neon.h>

// Avoid using i64 types for vld1q_lane and vst1q_lane Neon intrinsics with
// <2 x i64> vectors to avoid poor code for i64 in the backend.
void t1(uint64_t *src, uint8_t *dst) {
// CHECK: @t1
  uint64x2_t q = vld1q_u64(src);
// CHECK: call <2 x i64> @llvm.arm.neon.vld1.v2i64.p0
  vst1q_lane_u64(dst, q, 1);
// CHECK: shufflevector <2 x i64>
// CHECK: call void @llvm.arm.neon.vst1.p0.v1i64
}

void t2(uint64_t *src1, uint8_t *src2, uint64x2_t *dst) {
// CHECK: @t2
    uint64x2_t q = vld1q_u64(src1);
// CHECK: call <2 x i64> @llvm.arm.neon.vld1.v2i64.p0
    q = vld1q_lane_u64(src2, q, 0);
// CHECK: shufflevector <2 x i64>
// CHECK: call <1 x i64> @llvm.arm.neon.vld1.v1i64.p0
// CHECK: shufflevector <1 x i64>
    *dst = q;
// CHECK: store <2 x i64>
}
