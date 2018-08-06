// RUN: %clang_cc1 -triple thumbv7-none-linux-gnueabihf \
// RUN:   -target-abi aapcs \
// RUN:   -target-cpu cortex-a7 \
// RUN:   -mfloat-abi hard \
// RUN:   -ffreestanding \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_fma_order(<2 x float> %accum, <2 x float> %lhs, <2 x float> %rhs) #0 {
// CHECK:   [[TMP6:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %lhs, <2 x float> %rhs, <2 x float> %accum) #2
// CHECK:   ret <2 x float> [[TMP6]]
float32x2_t test_fma_order(float32x2_t accum, float32x2_t lhs, float32x2_t rhs) {
  return vfma_f32(accum, lhs, rhs);
}

// CHECK-LABEL: define <4 x float> @test_fmaq_order(<4 x float> %accum, <4 x float> %lhs, <4 x float> %rhs) #0 {
// CHECK:   [[TMP6:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %lhs, <4 x float> %rhs, <4 x float> %accum) #2
// CHECK:   ret <4 x float> [[TMP6]]
float32x4_t test_fmaq_order(float32x4_t accum, float32x4_t lhs, float32x4_t rhs) {
  return vfmaq_f32(accum, lhs, rhs);
}

// CHECK-LABEL: define <2 x float> @test_vfma_n_f32(<2 x float> %a, <2 x float> %b, float %n) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <2 x float> [[VECINIT1_I]] to <8 x i8>
// CHECK:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %b, <2 x float> [[VECINIT1_I]], <2 x float> %a)
// CHECK:   ret <2 x float> [[TMP3]]
float32x2_t test_vfma_n_f32(float32x2_t a, float32x2_t b, float32_t n) {
  return vfma_n_f32(a, b, n);
}

// CHECK-LABEL: define <4 x float> @test_vfmaq_n_f32(<4 x float> %a, <4 x float> %b, float %n) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %n, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %n, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %n, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %n, i32 3
// CHECK:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <4 x float> [[VECINIT3_I]] to <16 x i8>
// CHECK:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %b, <4 x float> [[VECINIT3_I]], <4 x float> %a)
// CHECK:   ret <4 x float> [[TMP3]]
float32x4_t test_vfmaq_n_f32(float32x4_t a, float32x4_t b, float32_t n) {
  return vfmaq_n_f32(a, b, n);
}
