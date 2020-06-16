// RUN: %clang_cc1 -triple aarch64-arm-none-eabi -target-feature +neon -target-feature +bf16 \
// RUN: -disable-O0-optnone -emit-llvm %s -o - | opt -S -mem2reg -instcombine | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: test_vbfdot_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <4 x bfloat> %a to <8 x i8>
// CHECK-NEXT  %1 = bitcast <4 x bfloat> %b to <8 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v8i8(<2 x float> %r, <8 x i8> %0, <8 x i8> %1)
// CHECK-NEXT  ret <2 x float> %vbfdot1.i
float32x2_t test_vbfdot_f32(float32x2_t r, bfloat16x4_t a, bfloat16x4_t b) {
  return vbfdot_f32(r, a, b);
}

// CHECK-LABEL: test_vbfdotq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %b to <16 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfdot1.i
float32x4_t test_vbfdotq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b){
  return vbfdotq_f32(r, a, b);
}

// CHECK-LABEL: test_vbfdot_lane_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <4 x bfloat> %b to <2 x float>
// CHECK-NEXT  %lane = shufflevector <2 x float> %0, <2 x float> undef, <2 x i32> zeroinitializer
// CHECK-NEXT  %1 = bitcast <4 x bfloat> %a to <8 x i8>
// CHECK-NEXT  %2 = bitcast <2 x float> %lane to <8 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v8i8(<2 x float> %r, <8 x i8> %1, <8 x i8> %2)
// CHECK-NEXT  ret <2 x float> %vbfdot1.i
float32x2_t test_vbfdot_lane_f32(float32x2_t r, bfloat16x4_t a, bfloat16x4_t b){
  return vbfdot_lane_f32(r, a, b, 0);
}

// CHECK-LABEL: test_vbfdotq_laneq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %b to <4 x float>
// CHECK-NEXT  %lane = shufflevector <4 x float> %0, <4 x float> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %2 = bitcast <4 x float> %lane to <16 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v16i8(<4 x float> %r, <16 x i8> %1, <16 x i8> %2)
// CHECK-NEXT  ret <4 x float> %vbfdot1.i
float32x4_t test_vbfdotq_laneq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfdotq_laneq_f32(r, a, b, 3);
}

// CHECK-LABEL: test_vbfdot_laneq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %b to <4 x float>
// CHECK-NEXT  %lane = shufflevector <4 x float> %0, <4 x float> undef, <2 x i32> <i32 3, i32 3>
// CHECK-NEXT  %1 = bitcast <4 x bfloat> %a to <8 x i8>
// CHECK-NEXT  %2 = bitcast <2 x float> %lane to <8 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <2 x float> @llvm.aarch64.neon.bfdot.v2f32.v8i8(<2 x float> %r, <8 x i8> %1, <8 x i8> %2)
// CHECK-NEXT  ret <2 x float> %vbfdot1.i
float32x2_t test_vbfdot_laneq_f32(float32x2_t r, bfloat16x4_t a, bfloat16x8_t b) {
  return vbfdot_laneq_f32(r, a, b, 3);
}

// CHECK-LABEL: test_vbfdotq_lane_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <4 x bfloat> %b to <2 x float>
// CHECK-NEXT  %lane = shufflevector <2 x float> %0, <2 x float> undef, <4 x i32> zeroinitializer
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %2 = bitcast <4 x float> %lane to <16 x i8>
// CHECK-NEXT  %vbfdot1.i = tail call <4 x float> @llvm.aarch64.neon.bfdot.v4f32.v16i8(<4 x float> %r, <16 x i8> %1, <16 x i8> %2)
// CHECK-NEXT  ret <4 x float> %vbfdot1.i
float32x4_t test_vbfdotq_lane_f32(float32x4_t r, bfloat16x8_t a, bfloat16x4_t b) {
  return vbfdotq_lane_f32(r, a, b, 0);
}

// CHECK-LABEL: test_vbfmmlaq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %b to <16 x i8>
// CHECK-NEXT  %vbfmmla1.i = tail call <4 x float> @llvm.aarch64.neon.bfmmla.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmmla1.i
float32x4_t test_vbfmmlaq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmmlaq_f32(r, a, b);
}

// CHECK-LABEL: test_vbfmlalbq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %b to <16 x i8>
// CHECK-NEXT  %vbfmlalb1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalb.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalb1.i
float32x4_t test_vbfmlalbq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmlalbq_f32(r, a, b);
}

// CHECK-LABEL: test_vbfmlaltq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %b to <16 x i8>
// CHECK-NEXT  %vbfmlalt1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalt.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalt1.i
float32x4_t test_vbfmlaltq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmlaltq_f32(r, a, b);
}

// CHECK-LABEL: test_vbfmlalbq_lane_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %vecinit35 = shufflevector <4 x bfloat> %b, <4 x bfloat> undef, <8 x i32> zeroinitializer
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
// CHECK-NEXT  %vbfmlalb1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalb.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalb1.i
float32x4_t test_vbfmlalbq_lane_f32(float32x4_t r, bfloat16x8_t a, bfloat16x4_t b) {
  return vbfmlalbq_lane_f32(r, a, b, 0);
}

// CHECK-LABEL: test_vbfmlalbq_laneq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %vecinit35 = shufflevector <8 x bfloat> %b, <8 x bfloat> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
// CHECK-NEXT  %vbfmlalb1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalb.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalb1.i
float32x4_t test_vbfmlalbq_laneq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmlalbq_laneq_f32(r, a, b, 3);
}

// CHECK-LABEL: test_vbfmlaltq_lane_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %vecinit35 = shufflevector <4 x bfloat> %b, <4 x bfloat> undef, <8 x i32> zeroinitializer
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
// CHECK-NEXT  %vbfmlalt1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalt.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalt1.i
float32x4_t test_vbfmlaltq_lane_f32(float32x4_t r, bfloat16x8_t a, bfloat16x4_t b) {
  return vbfmlaltq_lane_f32(r, a, b, 0);
}

// CHECK-LABEL: test_vbfmlaltq_laneq_f32
// CHECK-NEXT: entry:
// CHECK-NEXT  %vecinit35 = shufflevector <8 x bfloat> %b, <8 x bfloat> undef, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK-NEXT  %0 = bitcast <8 x bfloat> %a to <16 x i8>
// CHECK-NEXT  %1 = bitcast <8 x bfloat> %vecinit35 to <16 x i8>
// CHECK-NEXT  %vbfmlalt1.i = tail call <4 x float> @llvm.aarch64.neon.bfmlalt.v4f32.v16i8(<4 x float> %r, <16 x i8> %0, <16 x i8> %1)
// CHECK-NEXT  ret <4 x float> %vbfmlalt1.i
float32x4_t test_vbfmlaltq_laneq_f32(float32x4_t r, bfloat16x8_t a, bfloat16x8_t b) {
  return vbfmlaltq_laneq_f32(r, a, b, 3);
}
