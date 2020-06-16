// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-feature +neon -target-feature +bf16 \
// RUN:  -O2 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -triple armv8.6a-arm-none-eabi -target-feature +neon -target-feature +bf16 -mfloat-abi hard \
// RUN:  -O2 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK32

#include "arm_neon.h"

bfloat16x4_t test_vld1_bf16(bfloat16_t const *ptr) {
  return vld1_bf16(ptr);
}
// CHECK-LABEL: test_vld1_bf16
// CHECK64: %1 = load <4 x bfloat>, <4 x bfloat>* %0
// CHECK64-NEXT: ret <4 x bfloat> %1
// CHECK32: %1 = load <4 x bfloat>, <4 x bfloat>* %0, align 2
// CHECK32-NEXT: ret <4 x bfloat> %1

bfloat16x8_t test_vld1q_bf16(bfloat16_t const *ptr) {
  return vld1q_bf16(ptr);
}
// CHECK-LABEL: test_vld1q_bf16
// CHECK64: %1 = load <8 x bfloat>, <8 x bfloat>* %0
// CHECK64-NEXT: ret <8 x bfloat> %1
// CHECK32: %1 = load <8 x bfloat>, <8 x bfloat>* %0, align 2
// CHECK32-NEXT: ret <8 x bfloat> %1

bfloat16x4_t test_vld1_lane_bf16(bfloat16_t const *ptr, bfloat16x4_t src) {
  return vld1_lane_bf16(ptr, src, 0);
}
// CHECK-LABEL: test_vld1_lane_bf16
// CHECK64: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK64-NEXT: %vld1_lane = insertelement <4 x bfloat> %src, bfloat %0, i32 0
// CHECK64-NEXT: ret <4 x bfloat> %vld1_lane
// CHECK32: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK32-NEXT: %vld1_lane = insertelement <4 x bfloat> %src, bfloat %0, i32 0
// CHECK32-NEXT: ret <4 x bfloat> %vld1_lane

bfloat16x8_t test_vld1q_lane_bf16(bfloat16_t const *ptr, bfloat16x8_t src) {
  return vld1q_lane_bf16(ptr, src, 7);
}
// CHECK-LABEL: test_vld1q_lane_bf16
// CHECK64: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK64-NEXT: %vld1_lane = insertelement <8 x bfloat> %src, bfloat %0, i32 7
// CHECK64-NEXT: ret <8 x bfloat> %vld1_lane
// CHECK32: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK32-NEXT: %vld1_lane = insertelement <8 x bfloat> %src, bfloat %0, i32 7
// CHECK32-NEXT: ret <8 x bfloat> %vld1_lane

bfloat16x4_t test_vld1_dup_bf16(bfloat16_t const *ptr) {
  return vld1_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld1_dup_bf16
// CHECK64: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK64-NEXT: %1 = insertelement <4 x bfloat> undef, bfloat %0, i32 0
// CHECK64-NEXT: %lane = shufflevector <4 x bfloat> %1, <4 x bfloat> undef, <4 x i32> zeroinitializer
// CHECK64-NEXT: ret <4 x bfloat> %lane
// CHECK32: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK32-NEXT: %1 = insertelement <4 x bfloat> undef, bfloat %0, i32 0
// CHECK32-NEXT: %lane = shufflevector <4 x bfloat> %1, <4 x bfloat> undef, <4 x i32> zeroinitializer
// CHECK32-NEXT: ret <4 x bfloat> %lane

bfloat16x4x2_t test_vld1_bf16_x2(bfloat16_t const *ptr) {
  return vld1_bf16_x2(ptr);
}
// CHECK-LABEL: test_vld1_bf16_x2
// CHECK64: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x2.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld1x2.v4bf16.p0bf16(bfloat* %ptr)

bfloat16x8x2_t test_vld1q_bf16_x2(bfloat16_t const *ptr) {
  return vld1q_bf16_x2(ptr);
}
// CHECK-LABEL: test_vld1q_bf16_x2
// CHECK64: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x2.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld1x2.v8bf16.p0bf16(bfloat* %ptr)

bfloat16x4x3_t test_vld1_bf16_x3(bfloat16_t const *ptr) {
  return vld1_bf16_x3(ptr);
}
// CHECK-LABEL: test_vld1_bf16_x3
// CHECK64: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x3.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld1x3.v4bf16.p0bf16(bfloat* %ptr)

bfloat16x8x3_t test_vld1q_bf16_x3(bfloat16_t const *ptr) {
  return vld1q_bf16_x3(ptr);
}
// CHECK-LABEL: test_vld1q_bf16_x3
// CHECK64: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x3.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld1x3.v8bf16.p0bf16(bfloat* %ptr)

bfloat16x4x4_t test_vld1_bf16_x4(bfloat16_t const *ptr) {
  return vld1_bf16_x4(ptr);
}
// CHECK-LABEL: test_vld1_bf16_x4
// CHECK64: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x4.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld1x4.v4bf16.p0bf16(bfloat* %ptr)

bfloat16x8x4_t test_vld1q_bf16_x4(bfloat16_t const *ptr) {
  return vld1q_bf16_x4(ptr);
}
// CHECK-LABEL: test_vld1q_bf16_x4
// CHECK64: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x4.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld1x4.v8bf16.p0bf16(bfloat* %ptr)

bfloat16x8_t test_vld1q_dup_bf16(bfloat16_t const *ptr) {
  return vld1q_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld1q_dup_bf16
// CHECK64: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK64-NEXT: %1 = insertelement <8 x bfloat> undef, bfloat %0, i32 0
// CHECK64-NEXT: %lane = shufflevector <8 x bfloat> %1, <8 x bfloat> undef, <8 x i32> zeroinitializer
// CHECK64-NEXT: ret <8 x bfloat> %lane
// CHECK32: %0 = load bfloat, bfloat* %ptr, align 2
// CHECK32-NEXT: %1 = insertelement <8 x bfloat> undef, bfloat %0, i32 0
// CHECK32-NEXT: %lane = shufflevector <8 x bfloat> %1, <8 x bfloat> undef, <8 x i32> zeroinitializer
// CHECK32-NEXT: ret <8 x bfloat> %lane

bfloat16x4x2_t test_vld2_bf16(bfloat16_t const *ptr) {
  return vld2_bf16(ptr);
}
// CHECK-LABEL: test_vld2_bf16
// CHECK64:  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
// CHECK64-NEXT:  %vld2 = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2.v4bf16.p0v4bf16(<4 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld2_v = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld2.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x2_t test_vld2q_bf16(bfloat16_t const *ptr) {
  return vld2q_bf16(ptr);
}
// CHECK-LABEL: test_vld2q_bf16
// CHECK64: %0 = bitcast bfloat* %ptr to <8 x bfloat>*
// CHECK64-NEXT: %vld2 = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2.v8bf16.p0v8bf16(<8 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld2q_v = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld2.v8bf16.p0i8(i8* %0, i32 2)

bfloat16x4x2_t test_vld2_lane_bf16(bfloat16_t const *ptr, bfloat16x4x2_t src) {
  return vld2_lane_bf16(ptr, src, 1);
}
// CHECK-LABEL: test_vld2_lane_bf16
// CHECK64: %vld2_lane = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, i64 1, i8* %0)
// CHECK32: %vld2_lane_v = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld2lane.v4bf16.p0i8(i8* %2, <4 x bfloat> %0, <4 x bfloat> %1, i32 1, i32 2)

bfloat16x8x2_t test_vld2q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x2_t src) {
  return vld2q_lane_bf16(ptr, src, 7);
}
// CHECK-LABEL: test_vld2q_lane_bf16
// CHECK64: %vld2_lane = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, i64 7, i8* %0)
// CHECK32: %vld2q_lane_v = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld2lane.v8bf16.p0i8(i8* %2, <8 x bfloat> %0, <8 x bfloat> %1, i32 7, i32 2)

bfloat16x4x3_t test_vld3_bf16(bfloat16_t const *ptr) {
  return vld3_bf16(ptr);
}
// CHECK-LABEL: test_vld3_bf16
// CHECK64: %vld3 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3.v4bf16.p0v4bf16(<4 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld3_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld3.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x3_t test_vld3q_bf16(bfloat16_t const *ptr) {
  return vld3q_bf16(ptr);
}
// CHECK-LABEL: test_vld3q_bf16
// CHECK64: %vld3 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3.v8bf16.p0v8bf16(<8 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld3q_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld3.v8bf16.p0i8(i8* %0, i32 2)

bfloat16x4x3_t test_vld3_lane_bf16(bfloat16_t const *ptr, bfloat16x4x3_t src) {
  return vld3_lane_bf16(ptr, src, 1);
}
// CHECK-LABEL: test_vld3_lane_bf16
// CHECK64: %vld3_lane = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, <4 x bfloat> %src.coerce.fca.2.extract, i64 1, i8* %0)
// CHECK32: %3 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld3_lane_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld3lane.v4bf16.p0i8(i8* %3, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, i32 1, i32 2)

bfloat16x8x3_t test_vld3q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x3_t src) {
  return vld3q_lane_bf16(ptr, src, 7);
  // return vld3q_lane_bf16(ptr, src, 8);
}
// CHECK-LABEL: test_vld3q_lane_bf16
// CHECK64: %vld3_lane = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, <8 x bfloat> %src.coerce.fca.2.extract, i64 7, i8* %0)
// CHECK32: %3 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld3q_lane_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld3lane.v8bf16.p0i8(i8* %3, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, i32 7, i32 2)

bfloat16x4x4_t test_vld4_bf16(bfloat16_t const *ptr) {
  return vld4_bf16(ptr);
}
// CHECK-LABEL: test_vld4_bf16
// CHECK64: %vld4 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4.v4bf16.p0v4bf16(<4 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld4_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld4.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x4_t test_vld4q_bf16(bfloat16_t const *ptr) {
  return vld4q_bf16(ptr);
}
// CHECK-LABEL: test_vld4q_bf16
// CHECK64: %vld4 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4.v8bf16.p0v8bf16(<8 x bfloat>* %0)
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld4q_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld4.v8bf16.p0i8(i8* %0, i32 2)

bfloat16x4x4_t test_vld4_lane_bf16(bfloat16_t const *ptr, bfloat16x4x4_t src) {
  return vld4_lane_bf16(ptr, src, 1);
}
// CHECK-LABEL: test_vld4_lane_bf16
// CHECK64: %vld4_lane = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, <4 x bfloat> %src.coerce.fca.2.extract, <4 x bfloat> %src.coerce.fca.3.extract, i64 1, i8* %0)
// CHECK32: %4 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld4_lane_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld4lane.v4bf16.p0i8(i8* %4, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, <4 x bfloat> %3, i32 1, i32 2)

bfloat16x8x4_t test_vld4q_lane_bf16(bfloat16_t const *ptr, bfloat16x8x4_t src) {
  return vld4q_lane_bf16(ptr, src, 7);
}
// CHECK-LABEL: test_vld4q_lane_bf16
// CHECK64: %vld4_lane = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, <8 x bfloat> %src.coerce.fca.2.extract, <8 x bfloat> %src.coerce.fca.3.extract, i64 7, i8* %0)
// CHECK32: %4 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: %vld4q_lane_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld4lane.v8bf16.p0i8(i8* %4, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, <8 x bfloat> %3, i32 7, i32 2)

bfloat16x4x2_t test_vld2_dup_bf16(bfloat16_t const *ptr) {
  return vld2_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld2_dup_bf16
// CHECK64: %vld2 = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2r.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld2_dup_v = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld2dup.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x2_t test_vld2q_dup_bf16(bfloat16_t const *ptr) {
  return vld2q_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld2q_dup_bf16
// CHECK64: %vld2 = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2r.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld2q_dup_v = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld2dup.v8bf16.p0i8(i8* %0, i32 2)

bfloat16x4x3_t test_vld3_dup_bf16(bfloat16_t const *ptr) {
  return vld3_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld3_dup_bf16
// CHECK64: %vld3 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3r.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld3_dup_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld3dup.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x3_t test_vld3q_dup_bf16(bfloat16_t const *ptr) {
  return vld3q_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld3q_dup_bf16
// CHECK64: %vld3 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3r.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld3q_dup_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld3dup.v8bf16.p0i8(i8* %0, i32 2)

bfloat16x4x4_t test_vld4_dup_bf16(bfloat16_t const *ptr) {
  return vld4_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld4_dup_bf16
// CHECK64: %vld4 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4r.v4bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld4_dup_v = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.arm.neon.vld4dup.v4bf16.p0i8(i8* %0, i32 2)

bfloat16x8x4_t test_vld4q_dup_bf16(bfloat16_t const *ptr) {
  return vld4q_dup_bf16(ptr);
}
// CHECK-LABEL: test_vld4q_dup_bf16
// CHECK64: %vld4 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4r.v8bf16.p0bf16(bfloat* %ptr)
// CHECK32: %vld4q_dup_v = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.arm.neon.vld4dup.v8bf16.p0i8(i8* %0, i32 2)

void test_vst1_bf16(bfloat16_t *ptr, bfloat16x4_t val) {
  vst1_bf16(ptr, val);
}
// CHECK-LABEL: test_vst1_bf16
// CHECK64: %0 = bitcast bfloat* %ptr to <4 x bfloat>*
// CHECK64-NEXT: store <4 x bfloat> %val, <4 x bfloat>* %0, align 2
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: tail call void @llvm.arm.neon.vst1.p0i8.v4bf16(i8* %0, <4 x bfloat> %val, i32 2)

void test_vst1q_bf16(bfloat16_t *ptr, bfloat16x8_t val) {
  vst1q_bf16(ptr, val);
}
// CHECK-LABEL: test_vst1q_bf16
// CHECK64: %0 = bitcast bfloat* %ptr to <8 x bfloat>*
// CHECK64-NEXT: store <8 x bfloat> %val, <8 x bfloat>* %0, align 2
// CHECK32: %0 = bitcast bfloat* %ptr to i8*
// CHECK32-NEXT: tail call void @llvm.arm.neon.vst1.p0i8.v8bf16(i8* %0, <8 x bfloat> %val, i32 2)

void test_vst1_lane_bf16(bfloat16_t *ptr, bfloat16x4_t val) {
  vst1_lane_bf16(ptr, val, 1);
}
// CHECK-LABEL: test_vst1_lane_bf16
// CHECK64: %0 = extractelement <4 x bfloat> %val, i32 1
// CHECK64-NEXT: store bfloat %0, bfloat* %ptr, align 2
// CHECK32: %0 = extractelement <4 x bfloat> %val, i32 1
// CHECK32-NEXT: store bfloat %0, bfloat* %ptr, align 2

void test_vst1q_lane_bf16(bfloat16_t *ptr, bfloat16x8_t val) {
  vst1q_lane_bf16(ptr, val, 7);
}
// CHECK-LABEL: test_vst1q_lane_bf16
// CHECK64: %0 = extractelement <8 x bfloat> %val, i32 7
// CHECK64-NEXT: store bfloat %0, bfloat* %ptr, align 2
// CHECK32: %0 = extractelement <8 x bfloat> %val, i32 7
// CHECK32-NEXT: store bfloat %0, bfloat* %ptr, align 2

void test_vst1_bf16_x2(bfloat16_t *ptr, bfloat16x4x2_t val) {
  vst1_bf16_x2(ptr, val);
}
// CHECK-LABEL: test_vst1_bf16_x2
// CHECK64: tail call void @llvm.aarch64.neon.st1x2.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x2.p0bf16.v4bf16(bfloat* %ptr, <4 x bfloat> %0, <4 x bfloat> %1)

void test_vst1q_bf16_x2(bfloat16_t *ptr, bfloat16x8x2_t val) {
  vst1q_bf16_x2(ptr, val);
}
// CHECK-LABEL: test_vst1q_bf16_x2
// CHECK64: tail call void @llvm.aarch64.neon.st1x2.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x2.p0bf16.v8bf16(bfloat* %ptr, <8 x bfloat> %0, <8 x bfloat> %1)

void test_vst1_bf16_x3(bfloat16_t *ptr, bfloat16x4x3_t val) {
  vst1_bf16_x3(ptr, val);
}
// CHECK-LABEL: test_vst1_bf16_x3
// CHECK64: tail call void @llvm.aarch64.neon.st1x3.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x3.p0bf16.v4bf16(bfloat* %ptr, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2)

void test_vst1q_bf16_x3(bfloat16_t *ptr, bfloat16x8x3_t val) {
  vst1q_bf16_x3(ptr, val);
}
// CHECK-LABEL: test_vst1q_bf16_x3
// CHECK64: tail call void @llvm.aarch64.neon.st1x3.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x3.p0bf16.v8bf16(bfloat* %ptr, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2)

void test_vst1_bf16_x4(bfloat16_t *ptr, bfloat16x4x4_t val) {
  vst1_bf16_x4(ptr, val);
}
// CHECK-LABEL: test_vst1_bf16_x4
// CHECK64: tail call void @llvm.aarch64.neon.st1x4.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x4.p0bf16.v4bf16(bfloat* %ptr, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, <4 x bfloat> %3)

void test_vst1q_bf16_x4(bfloat16_t *ptr, bfloat16x8x4_t val) {
  vst1q_bf16_x4(ptr, val);
}
// CHECK-LABEL: test_vst1q_bf16_x4
// CHECK64: tail call void @llvm.aarch64.neon.st1x4.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, bfloat* %ptr)
// CHECK32: tail call void @llvm.arm.neon.vst1x4.p0bf16.v8bf16(bfloat* %ptr, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, <8 x bfloat> %3)

void test_vst2_bf16(bfloat16_t *ptr, bfloat16x4x2_t val) {
  vst2_bf16(ptr, val);
}
// CHECK-LABEL: test_vst2_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st2.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst2.p0i8.v4bf16(i8* %2, <4 x bfloat> %0, <4 x bfloat> %1, i32 2)

void test_vst2q_bf16(bfloat16_t *ptr, bfloat16x8x2_t val) {
  vst2q_bf16(ptr, val);
}
// CHECK-LABEL: test_vst2q_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st2.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst2.p0i8.v8bf16(i8* %2, <8 x bfloat> %0, <8 x bfloat> %1, i32 2)

void test_vst2_lane_bf16(bfloat16_t *ptr, bfloat16x4x2_t val) {
  vst2_lane_bf16(ptr, val, 1);
}
// CHECK-LABEL: test_vst2_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st2lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, i64 1, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst2lane.p0i8.v4bf16(i8* %2, <4 x bfloat> %0, <4 x bfloat> %1, i32 1, i32 2)

void test_vst2q_lane_bf16(bfloat16_t *ptr, bfloat16x8x2_t val) {
  vst2q_lane_bf16(ptr, val, 7);
}
// CHECK-LABEL: test_vst2q_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st2lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, i64 7, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst2lane.p0i8.v8bf16(i8* %2, <8 x bfloat> %0, <8 x bfloat> %1, i32 7, i32 2)

void test_vst3_bf16(bfloat16_t *ptr, bfloat16x4x3_t val) {
  vst3_bf16(ptr, val);
}
// CHECK-LABEL: test_vst3_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st3.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst3.p0i8.v4bf16(i8* %3, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, i32 2)

void test_vst3q_bf16(bfloat16_t *ptr, bfloat16x8x3_t val) {
  vst3q_bf16(ptr, val);
}
// CHECK-LABEL: test_vst3q_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st3.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, i8* %0)
// CHECK32:  tail call void @llvm.arm.neon.vst3.p0i8.v8bf16(i8* %3, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, i32 2)

void test_vst3_lane_bf16(bfloat16_t *ptr, bfloat16x4x3_t val) {
  vst3_lane_bf16(ptr, val, 1);
}
// CHECK-LABEL: test_vst3_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st3lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, i64 1, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst3lane.p0i8.v4bf16(i8* %3, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, i32 1, i32 2)

void test_vst3q_lane_bf16(bfloat16_t *ptr, bfloat16x8x3_t val) {
  vst3q_lane_bf16(ptr, val, 7);
}
// CHECK-LABEL: test_vst3q_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st3lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, i64 7, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst3lane.p0i8.v8bf16(i8* %3, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, i32 7, i32 2)

void test_vst4_bf16(bfloat16_t *ptr, bfloat16x4x4_t val) {
  vst4_bf16(ptr, val);
}
// CHECK-LABEL: test_vst4_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st4.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst4.p0i8.v4bf16(i8* %4, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, <4 x bfloat> %3, i32 2)

void test_vst4q_bf16(bfloat16_t *ptr, bfloat16x8x4_t val) {
  vst4q_bf16(ptr, val);
}
// CHECK-LABEL: test_vst4q_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st4.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst4.p0i8.v8bf16(i8* %4, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, <8 x bfloat> %3, i32 2)

void test_vst4_lane_bf16(bfloat16_t *ptr, bfloat16x4x4_t val) {
  vst4_lane_bf16(ptr, val, 1);
}
// CHECK-LABEL: test_vst4_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st4lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, i64 1, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst4lane.p0i8.v4bf16(i8* %4, <4 x bfloat> %0, <4 x bfloat> %1, <4 x bfloat> %2, <4 x bfloat> %3, i32 1, i32 2)

void test_vst4q_lane_bf16(bfloat16_t *ptr, bfloat16x8x4_t val) {
  vst4q_lane_bf16(ptr, val, 7);
}
// CHECK-LABEL: test_vst4q_lane_bf16
// CHECK64: tail call void @llvm.aarch64.neon.st4lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, i64 7, i8* %0)
// CHECK32: tail call void @llvm.arm.neon.vst4lane.p0i8.v8bf16(i8* %4, <8 x bfloat> %0, <8 x bfloat> %1, <8 x bfloat> %2, <8 x bfloat> %3, i32 7, i32 2)
