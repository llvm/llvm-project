// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -S -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

// CHECK-LABEL: define <2 x float> @test_vmla_n_f32(<2 x float> %a, <2 x float> %b, float %c) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %c, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %b, [[VECINIT1_I]]
// CHECK:   [[ADD_I:%.*]] = fadd <2 x float> %a, [[MUL_I]]
// CHECK:   ret <2 x float> [[ADD_I]]
float32x2_t test_vmla_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  return vmla_n_f32(a, b, c);
}

// CHECK-LABEL: define <4 x float> @test_vmlaq_n_f32(<4 x float> %a, <4 x float> %b, float %c) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %c, i32 3
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %b, [[VECINIT3_I]]
// CHECK:   [[ADD_I:%.*]] = fadd <4 x float> %a, [[MUL_I]]
// CHECK:   ret <4 x float> [[ADD_I]]
float32x4_t test_vmlaq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  return vmlaq_n_f32(a, b, c);
}

// CHECK-LABEL: define <2 x double> @test_vmlaq_n_f64(<2 x double> %a, <2 x double> %b, double %c) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %c, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %b, [[VECINIT1_I]]
// CHECK:   [[ADD_I:%.*]] = fadd <2 x double> %a, [[MUL_I]]
// CHECK:   ret <2 x double> [[ADD_I]]
float64x2_t test_vmlaq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  return vmlaq_n_f64(a, b, c);
}

// CHECK-LABEL: define <4 x float> @test_vmlsq_n_f32(<4 x float> %a, <4 x float> %b, float %c) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <4 x float> undef, float %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <4 x float> [[VECINIT_I]], float %c, i32 1
// CHECK:   [[VECINIT2_I:%.*]] = insertelement <4 x float> [[VECINIT1_I]], float %c, i32 2
// CHECK:   [[VECINIT3_I:%.*]] = insertelement <4 x float> [[VECINIT2_I]], float %c, i32 3
// CHECK:   [[MUL_I:%.*]] = fmul <4 x float> %b, [[VECINIT3_I]]
// CHECK:   [[SUB_I:%.*]] = fsub <4 x float> %a, [[MUL_I]]
// CHECK:   ret <4 x float> [[SUB_I]]
float32x4_t test_vmlsq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  return vmlsq_n_f32(a, b, c);
}

// CHECK-LABEL: define <2 x float> @test_vmls_n_f32(<2 x float> %a, <2 x float> %b, float %c) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x float> undef, float %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x float> [[VECINIT_I]], float %c, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x float> %b, [[VECINIT1_I]]
// CHECK:   [[SUB_I:%.*]] = fsub <2 x float> %a, [[MUL_I]]
// CHECK:   ret <2 x float> [[SUB_I]]
float32x2_t test_vmls_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  return vmls_n_f32(a, b, c);
}

// CHECK-LABEL: define <2 x double> @test_vmlsq_n_f64(<2 x double> %a, <2 x double> %b, double %c) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %c, i32 1
// CHECK:   [[MUL_I:%.*]] = fmul <2 x double> %b, [[VECINIT1_I]]
// CHECK:   [[SUB_I:%.*]] = fsub <2 x double> %a, [[MUL_I]]
// CHECK:   ret <2 x double> [[SUB_I]]
float64x2_t test_vmlsq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  return vmlsq_n_f64(a, b, c);
}

// CHECK-LABEL: define <2 x float> @test_vmla_lane_f32_0(<2 x float> %a, <2 x float> %b, <2 x float> %v) #0 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[ADD]]
float32x2_t test_vmla_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vmla_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: define <4 x float> @test_vmlaq_lane_f32_0(<4 x float> %a, <4 x float> %b, <2 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[ADD]]
float32x4_t test_vmlaq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vmlaq_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: define <2 x float> @test_vmla_laneq_f32_0(<2 x float> %a, <2 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[ADD]]
float32x2_t test_vmla_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vmla_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: define <4 x float> @test_vmlaq_laneq_f32_0(<4 x float> %a, <4 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[ADD]]
float32x4_t test_vmlaq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vmlaq_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: define <2 x float> @test_vmls_lane_f32_0(<2 x float> %a, <2 x float> %b, <2 x float> %v) #0 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[SUB]]
float32x2_t test_vmls_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vmls_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: define <4 x float> @test_vmlsq_lane_f32_0(<4 x float> %a, <4 x float> %b, <2 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[SUB]]
float32x4_t test_vmlsq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vmlsq_lane_f32(a, b, v, 0);
}

// CHECK-LABEL: define <2 x float> @test_vmls_laneq_f32_0(<2 x float> %a, <2 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[SUB]]
float32x2_t test_vmls_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vmls_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: define <4 x float> @test_vmlsq_laneq_f32_0(<4 x float> %a, <4 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> zeroinitializer
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[SUB]]
float32x4_t test_vmlsq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vmlsq_laneq_f32(a, b, v, 0);
}

// CHECK-LABEL: define <2 x float> @test_vmla_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) #0 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[ADD]]
float32x2_t test_vmla_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vmla_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: define <4 x float> @test_vmlaq_lane_f32(<4 x float> %a, <4 x float> %b, <2 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[ADD]]
float32x4_t test_vmlaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vmlaq_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: define <2 x float> @test_vmla_laneq_f32(<2 x float> %a, <2 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[ADD]]
float32x2_t test_vmla_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vmla_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: define <4 x float> @test_vmlaq_laneq_f32(<4 x float> %a, <4 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[ADD:%.*]] = fadd <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[ADD]]
float32x4_t test_vmlaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vmlaq_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: define <2 x float> @test_vmls_lane_f32(<2 x float> %a, <2 x float> %b, <2 x float> %v) #0 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <2 x i32> <i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[SUB]]
float32x2_t test_vmls_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  return vmls_lane_f32(a, b, v, 1);
}

// CHECK-LABEL: define <4 x float> @test_vmlsq_lane_f32(<4 x float> %a, <4 x float> %b, <2 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <2 x float> %v, <2 x float> %v, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[SUB]]
float32x4_t test_vmlsq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  return vmlsq_lane_f32(a, b, v, 1);
}
// CHECK-LABEL: define <2 x float> @test_vmls_laneq_f32(<2 x float> %a, <2 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> <i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <2 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <2 x float> %a, [[MUL]]
// CHECK:   ret <2 x float> [[SUB]]
float32x2_t test_vmls_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  return vmls_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: define <4 x float> @test_vmlsq_laneq_f32(<4 x float> %a, <4 x float> %b, <4 x float> %v) #1 {
// CHECK:   [[SHUFFLE:%.*]] = shufflevector <4 x float> %v, <4 x float> %v, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   [[MUL:%.*]] = fmul <4 x float> %b, [[SHUFFLE]]
// CHECK:   [[SUB:%.*]] = fsub <4 x float> %a, [[MUL]]
// CHECK:   ret <4 x float> [[SUB]]
float32x4_t test_vmlsq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  return vmlsq_laneq_f32(a, b, v, 3);
}

// CHECK-LABEL: define <2 x double> @test_vfmaq_n_f64(<2 x double> %a, <2 x double> %b, double %c) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %c, i32 1
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %b, <2 x double> [[VECINIT1_I]], <2 x double> %a)
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmaq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  return vfmaq_n_f64(a, b, c);
}

// CHECK-LABEL: define <2 x double> @test_vfmsq_n_f64(<2 x double> %a, <2 x double> %b, double %c) #1 {
// CHECK:   [[SUB_I:%.*]] = fneg <2 x double> %b
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x double> undef, double %c, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x double> [[VECINIT_I]], double %c, i32 1
// CHECK:   [[TMP6:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[SUB_I]], <2 x double> [[VECINIT1_I]], <2 x double> %a) #3
// CHECK:   ret <2 x double> [[TMP6]]
float64x2_t test_vfmsq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  return vfmsq_n_f64(a, b, c);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"
