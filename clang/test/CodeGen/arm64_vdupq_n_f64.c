// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -fallow-half-arguments-and-returns -S -o - -disable-O0-optnone -emit-llvm %s | opt -S -mem2reg | FileCheck %s

#include <arm_neon.h>

// vdupq_n_f64 -> dup.2d v0, v0[0]
//
// CHECK-LABEL: define <2 x double> @test_vdupq_n_f64(double %w) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %w, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %w, i32 1
// CHECK:   ret <2 x double> [[VECINIT1_I]]
float64x2_t test_vdupq_n_f64(float64_t w) {
    return vdupq_n_f64(w);
}

// might as well test this while we're here
// vdupq_n_f32 -> dup.4s v0, v0[0]
// CHECK-LABEL: define <4 x float> @test_vdupq_n_f32(float %w) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %w, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %w, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %w, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %w, i32 3
// CHECK:   ret <4 x float> [[VECINIT3_I]]
float32x4_t test_vdupq_n_f32(float32_t w) {
    return vdupq_n_f32(w);
}

// vdupq_lane_f64 -> dup.2d v0, v0[0]
// this was in <rdar://problem/11778405>, but had already been implemented,
// test anyway
// CHECK-LABEL: define <2 x double> @test_vdupq_lane_f64(<1 x double> %V) #0 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <1 x double> %V, <1 x double> %V, <2 x i32> zeroinitializer
// CHECK:   ret <2 x double> [[SHUFFLE]]
float64x2_t test_vdupq_lane_f64(float64x1_t V) {
    return vdupq_lane_f64(V, 0);
}

// vmovq_n_f64 -> dup Vd.2d,X0
// this wasn't in <rdar://problem/11778405>, but it was between the vdups
// CHECK-LABEL: define <2 x double> @test_vmovq_n_f64(double %w) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %w, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %w, i32 1
// CHECK:   ret <2 x double> [[VECINIT1_I]]
float64x2_t test_vmovq_n_f64(float64_t w) {
  return vmovq_n_f64(w);
}

// CHECK-LABEL: define <4 x half> @test_vmov_n_f16(half* %a1) #0 {
// CHECK:   [[TMP0:%.*]] = load half, half* %a1, align 2
// CHECK:   [[VECINIT:%.*]] = insertelement <4 x half> undef, half [[TMP0]], i32 0
// CHECK:   [[VECINIT1:%.*]] = insertelement <4 x half> [[VECINIT]], half [[TMP0]], i32 1
// CHECK:   [[VECINIT2:%.*]] = insertelement <4 x half> [[VECINIT1]], half [[TMP0]], i32 2
// CHECK:   [[VECINIT3:%.*]] = insertelement <4 x half> [[VECINIT2]], half [[TMP0]], i32 3
// CHECK:   ret <4 x half> [[VECINIT3]]
float16x4_t test_vmov_n_f16(float16_t *a1) {
  return vmov_n_f16(*a1);
}

/*
float64x1_t test_vmov_n_f64(float64_t a1) {
  return vmov_n_f64(a1);
}
*/

// CHECK-LABEL: define <8 x half> @test_vmovq_n_f16(half* %a1) #0 {
// CHECK:   [[TMP0:%.*]] = load half, half* %a1, align 2
// CHECK:   [[VECINIT:%.*]] = insertelement <8 x half> undef, half [[TMP0]], i32 0
// CHECK:   [[VECINIT1:%.*]] = insertelement <8 x half> [[VECINIT]], half [[TMP0]], i32 1
// CHECK:   [[VECINIT2:%.*]] = insertelement <8 x half> [[VECINIT1]], half [[TMP0]], i32 2
// CHECK:   [[VECINIT3:%.*]] = insertelement <8 x half> [[VECINIT2]], half [[TMP0]], i32 3
// CHECK:   [[VECINIT4:%.*]] = insertelement <8 x half> [[VECINIT3]], half [[TMP0]], i32 4
// CHECK:   [[VECINIT5:%.*]] = insertelement <8 x half> [[VECINIT4]], half [[TMP0]], i32 5
// CHECK:   [[VECINIT6:%.*]] = insertelement <8 x half> [[VECINIT5]], half [[TMP0]], i32 6
// CHECK:   [[VECINIT7:%.*]] = insertelement <8 x half> [[VECINIT6]], half [[TMP0]], i32 7
// CHECK:   ret <8 x half> [[VECINIT7]]
float16x8_t test_vmovq_n_f16(float16_t *a1) {
  return vmovq_n_f16(*a1);
}

