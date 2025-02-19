// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +bf16 \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +bf16 \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test mimics clang/test/CodeGen/AArch64/bf16-getset-intrinsics.c, which eventually
// CIR shall be able to support fully. Since this is going to take some time to converge,
// the unsupported/NYI code is commented out, so that we can incrementally improve this.
// The NYI filecheck used contains the LLVM output from OG codegen that should guide the
// correct result when implementing this into the CIR pipeline.

#include <arm_neon.h>

// CHECK-LABEL: @test_vcreate_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast i64 [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
// bfloat16x4_t test_vcreate_bf16(uint64_t a) {
//   return vcreate_bf16(a);
// }

// CHECK-LABEL: @test_vdup_n_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VECINIT_I:%.*]] = insertelement <4 x bfloat> poison, bfloat [[V:%.*]], i32 0
// CHECK-NEXT:    [[VECINIT1_I:%.*]] = insertelement <4 x bfloat> [[VECINIT_I]], bfloat [[V]], i32 1
// CHECK-NEXT:    [[VECINIT2_I:%.*]] = insertelement <4 x bfloat> [[VECINIT1_I]], bfloat [[V]], i32 2
// CHECK-NEXT:    [[VECINIT3_I:%.*]] = insertelement <4 x bfloat> [[VECINIT2_I]], bfloat [[V]], i32 3
// CHECK-NEXT:    ret <4 x bfloat> [[VECINIT3_I]]
//
// bfloat16x4_t test_vdup_n_bf16(bfloat16_t v) {
//   return vdup_n_bf16(v);
// }

// CHECK-LABEL: @test_vdupq_n_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VECINIT_I:%.*]] = insertelement <8 x bfloat> poison, bfloat [[V:%.*]], i32 0
// CHECK-NEXT:    [[VECINIT1_I:%.*]] = insertelement <8 x bfloat> [[VECINIT_I]], bfloat [[V]], i32 1
// CHECK-NEXT:    [[VECINIT2_I:%.*]] = insertelement <8 x bfloat> [[VECINIT1_I]], bfloat [[V]], i32 2
// CHECK-NEXT:    [[VECINIT3_I:%.*]] = insertelement <8 x bfloat> [[VECINIT2_I]], bfloat [[V]], i32 3
// CHECK-NEXT:    [[VECINIT4_I:%.*]] = insertelement <8 x bfloat> [[VECINIT3_I]], bfloat [[V]], i32 4
// CHECK-NEXT:    [[VECINIT5_I:%.*]] = insertelement <8 x bfloat> [[VECINIT4_I]], bfloat [[V]], i32 5
// CHECK-NEXT:    [[VECINIT6_I:%.*]] = insertelement <8 x bfloat> [[VECINIT5_I]], bfloat [[V]], i32 6
// CHECK-NEXT:    [[VECINIT7_I:%.*]] = insertelement <8 x bfloat> [[VECINIT6_I]], bfloat [[V]], i32 7
// CHECK-NEXT:    ret <8 x bfloat> [[VECINIT7_I]]
//
// bfloat16x8_t test_vdupq_n_bf16(bfloat16_t v) {
//   return vdupq_n_bf16(v);
// }

// CHECK-LABEL: @test_vdup_lane_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[V:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x bfloat>
// CHECK-NEXT:    [[LANE:%.*]] = shufflevector <4 x bfloat> [[TMP1]], <4 x bfloat> [[TMP1]], <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK-NEXT:    ret <4 x bfloat> [[LANE]]
//
// bfloat16x4_t test_vdup_lane_bf16(bfloat16x4_t v) {
//   return vdup_lane_bf16(v, 1);
// }

// CHECK-LABEL: @test_vdupq_lane_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[V:%.*]] to <8 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x bfloat>
// CHECK-NEXT:    [[LANE:%.*]] = shufflevector <4 x bfloat> [[TMP1]], <4 x bfloat> [[TMP1]], <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
// CHECK-NEXT:    ret <8 x bfloat> [[LANE]]
//
// bfloat16x8_t test_vdupq_lane_bf16(bfloat16x4_t v) {
//   return vdupq_lane_bf16(v, 1);
// }

// CHECK-LABEL: @test_vdup_laneq_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[V:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x bfloat>
// CHECK-NEXT:    [[LANE:%.*]] = shufflevector <8 x bfloat> [[TMP1]], <8 x bfloat> [[TMP1]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK-NEXT:    ret <4 x bfloat> [[LANE]]
//
// bfloat16x4_t test_vdup_laneq_bf16(bfloat16x8_t v) {
//   return vdup_laneq_bf16(v, 7);
// }

// CHECK-LABEL: @test_vdupq_laneq_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[V:%.*]] to <16 x i8>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x bfloat>
// CHECK-NEXT:    [[LANE:%.*]] = shufflevector <8 x bfloat> [[TMP1]], <8 x bfloat> [[TMP1]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK-NEXT:    ret <8 x bfloat> [[LANE]]
//
// bfloat16x8_t test_vdupq_laneq_bf16(bfloat16x8_t v) {
//   return vdupq_laneq_bf16(v, 7);
// }

// CHECK-LABEL: @test_vcombine_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <4 x bfloat> [[LOW:%.*]], <4 x bfloat> [[HIGH:%.*]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    ret <8 x bfloat> [[SHUFFLE_I]]
//
// bfloat16x8_t test_vcombine_bf16(bfloat16x4_t low, bfloat16x4_t high) {
//   return vcombine_bf16(low, high);
// }

// CHECK-LABEL: @test_vget_high_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x bfloat> [[A:%.*]], <8 x bfloat> [[A]], <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// CHECK-NEXT:    ret <4 x bfloat> [[SHUFFLE_I]]
//
// bfloat16x4_t test_vget_high_bf16(bfloat16x8_t a) {
//   return vget_high_bf16(a);
// }

// CHECK-LABEL: @test_vget_low_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[SHUFFLE_I:%.*]] = shufflevector <8 x bfloat> [[A:%.*]], <8 x bfloat> [[A]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// CHECK-NEXT:    ret <4 x bfloat> [[SHUFFLE_I]]
//
// bfloat16x4_t test_vget_low_bf16(bfloat16x8_t a) {
//   return vget_low_bf16(a);
// }

bfloat16_t test_vget_lane_bf16(bfloat16x4_t v) {
 return vget_lane_bf16(v, 1);

  // CIR-LABEL: vget_lane_bf16
  // CIR: [[TMP0:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[TMP1:%.*]] = cir.vec.extract {{.*}}[{{.*}} : !s32i] : !cir.vector<!cir.bf16 x 4>

  // LLVM-LABEL: test_vget_lane_bf16
  // LLVM-SAME: (<4 x bfloat> [[VEC:%.*]])
  // LLVM: [[VGET_LANE:%.*]] = extractelement <4 x bfloat> [[VEC]], i32 1
  // LLVM: ret bfloat [[VGET_LANE]]
}

// CHECK-LABEL: @test_vgetq_lane_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VGETQ_LANE:%.*]] = extractelement <8 x bfloat> [[V:%.*]], i32 7
// CHECK-NEXT:    ret bfloat [[VGETQ_LANE]]
//
// bfloat16_t test_vgetq_lane_bf16(bfloat16x8_t v) {
//   return vgetq_lane_bf16(v, 7);
// }

// CHECK-LABEL: @test_vset_lane_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VSET_LANE:%.*]] = insertelement <4 x bfloat> [[V:%.*]], bfloat [[A:%.*]], i32 1
// CHECK-NEXT:    ret <4 x bfloat> [[VSET_LANE]]
//
// bfloat16x4_t test_vset_lane_bf16(bfloat16_t a, bfloat16x4_t v) {
//   return vset_lane_bf16(a, v, 1);
// }

// CHECK-LABEL: @test_vsetq_lane_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[VSET_LANE:%.*]] = insertelement <8 x bfloat> [[V:%.*]], bfloat [[A:%.*]], i32 7
// CHECK-NEXT:    ret <8 x bfloat> [[VSET_LANE]]
//
// bfloat16x8_t test_vsetq_lane_bf16(bfloat16_t a, bfloat16x8_t v) {
//  return vsetq_lane_bf16(a, v, 7);
// }

bfloat16_t test_vduph_lane_bf16(bfloat16x4_t v) {
 return vduph_lane_bf16(v, 1);

  // CIR-LABEL: vduph_lane_bf16
  // CIR: [[TMP0:%.*]] = cir.const #cir.int<1> : !s32i
  // CIR: [[TMP1:%.*]] = cir.vec.extract {{.*}}[{{.*}} : !s32i] : !cir.vector<!cir.bf16 x 4>

  // LLVM-LABEL: test_vduph_lane_bf16
  // LLVM-SAME: (<4 x bfloat> [[VEC:%.*]])
  // LLVM: [[VGET_LANE:%.*]] = extractelement <4 x bfloat> [[VEC]], i32 1
  // LLVM: ret bfloat [[VGET_LANE]]
}
