// RUN: %clang_cc1 -O1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -S -o - -emit-llvm %s | FileCheck %s
// Test ARM64 SIMD fused multiply add intrinsics

#include <arm_neon.h>

float32x2_t test_vfma_f32(float32x2_t a1, float32x2_t a2, float32x2_t a3) {
  // CHECK: test_vfma_f32
  return vfma_f32(a1, a2, a3);
  // CHECK: llvm.fma.v2f32({{.*a2, .*a3, .*a1}})
  // CHECK-NEXT: ret
}

float32x4_t test_vfmaq_f32(float32x4_t a1, float32x4_t a2, float32x4_t a3) {
  // CHECK: test_vfmaq_f32
  return vfmaq_f32(a1, a2, a3);
  // CHECK: llvm.fma.v4f32({{.*a2, .*a3, .*a1}})
  // CHECK-NEXT: ret
}

float64x2_t test_vfmaq_f64(float64x2_t a1, float64x2_t a2, float64x2_t a3) {
  // CHECK: test_vfmaq_f64
  return vfmaq_f64(a1, a2, a3);
  // CHECK: llvm.fma.v2f64({{.*a2, .*a3, .*a1}})
  // CHECK-NEXT: ret
}

float32x2_t test_vfma_lane_f32(float32x2_t a1, float32x2_t a2, float32x2_t a3) {
  // CHECK: test_vfma_lane_f32
  return vfma_lane_f32(a1, a2, a3, 1);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: llvm.fma.v2f32(<2 x float> %a2, <2 x float> {{.*}}, <2 x float> %a1)
  // CHECK-NEXT: ret
}

float32x4_t test_vfmaq_lane_f32(float32x4_t a1, float32x4_t a2, float32x2_t a3) {
  // CHECK: test_vfmaq_lane_f32
  return vfmaq_lane_f32(a1, a2, a3, 1);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: llvm.fma.v4f32(<4 x float> %a2, <4 x float> {{.*}}, <4 x float> %a1)
  // CHECK-NEXT: ret
}

float64x2_t test_vfmaq_lane_f64(float64x2_t a1, float64x2_t a2, float64x1_t a3) {
  // CHECK: test_vfmaq_lane_f64
  return vfmaq_lane_f64(a1, a2, a3, 0);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: llvm.fma.v2f64(<2 x double> %a2, <2 x double> {{.*}}, <2 x double> %a1)
  // CHECK-NEXT: ret
}

float32x2_t test_vfma_n_f32(float32x2_t a1, float32x2_t a2, float32_t a3) {
  // CHECK: test_vfma_n_f32
  return vfma_n_f32(a1, a2, a3);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 0 (usually two insertelements)
  // CHECK: llvm.fma.v2f32
  // CHECK-NEXT: ret
}

float32x4_t test_vfmaq_n_f32(float32x4_t a1, float32x4_t a2, float32_t a3) {
  // CHECK: test_vfmaq_n_f32
  return vfmaq_n_f32(a1, a2, a3);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 0 (usually four insertelements)
  // CHECK: llvm.fma.v4f32
  // CHECK-NEXT: ret
}

float64x2_t test_vfmaq_n_f64(float64x2_t a1, float64x2_t a2, float64_t a3) {
  // CHECK: test_vfmaq_n_f64
  return vfmaq_n_f64(a1, a2, a3);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 0 (usually two insertelements)
  // CHECK: llvm.fma.v2f64
  // CHECK-NEXT: ret
}

float32x2_t test_vfms_f32(float32x2_t a1, float32x2_t a2, float32x2_t a3) {
  // CHECK: test_vfms_f32
  return vfms_f32(a1, a2, a3);
  // CHECK: [[NEG:%.*]] = fsub <2 x float> {{.*}}, %a2
  // CHECK: llvm.fma.v2f32(<2 x float> %a3, <2 x float> [[NEG]], <2 x float> %a1)
  // CHECK-NEXT: ret
}

float32x4_t test_vfmsq_f32(float32x4_t a1, float32x4_t a2, float32x4_t a3) {
  // CHECK: test_vfmsq_f32
  return vfmsq_f32(a1, a2, a3);
  // CHECK: [[NEG:%.*]] = fsub <4 x float> {{.*}}, %a2
  // CHECK: llvm.fma.v4f32(<4 x float> %a3, <4 x float> [[NEG]], <4 x float> %a1)
  // CHECK-NEXT: ret
}

float64x2_t test_vfmsq_f64(float64x2_t a1, float64x2_t a2, float64x2_t a3) {
  // CHECK: test_vfmsq_f64
  return vfmsq_f64(a1, a2, a3);
  // CHECK: [[NEG:%.*]] = fsub <2 x double> {{.*}}, %a2
  // CHECK: llvm.fma.v2f64(<2 x double> %a3, <2 x double> [[NEG]], <2 x double> %a1)
  // CHECK-NEXT: ret
}

float32x2_t test_vfms_lane_f32(float32x2_t a1, float32x2_t a2, float32x2_t a3) {
  // CHECK: test_vfms_lane_f32
  return vfms_lane_f32(a1, a2, a3, 1);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: [[NEG:%.*]] = fsub <2 x float> {{.*}}, %a3
  // CHECK: [[LANE:%.*]] = shufflevector <2 x float> [[NEG]]
  // CHECK: llvm.fma.v2f32(<2 x float> {{.*}}, <2 x float> [[LANE]], <2 x float> %a1)
  // CHECK-NEXT: ret
}

float32x4_t test_vfmsq_lane_f32(float32x4_t a1, float32x4_t a2, float32x2_t a3) {
  // CHECK: test_vfmsq_lane_f32
  return vfmsq_lane_f32(a1, a2, a3, 1);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: [[NEG:%.*]] = fsub <2 x float> {{.*}}, %a3
  // CHECK: [[LANE:%.*]] = shufflevector <2 x float> [[NEG]]
  // CHECK: llvm.fma.v4f32(<4 x float> {{.*}}, <4 x float> [[LANE]], <4 x float> %a1)
  // CHECK-NEXT: ret
}

float64x2_t test_vfmsq_lane_f64(float64x2_t a1, float64x2_t a2, float64x1_t a3) {
  // CHECK: test_vfmsq_lane_f64
  return vfmsq_lane_f64(a1, a2, a3, 0);
  // NB: the test below is deliberately lose, so that we don't depend too much
  // upon the exact IR used to select lane 1 (usually a shufflevector)
  // CHECK: [[NEG:%.*]] = fsub <1 x double> {{.*}}, %a3
  // CHECK: [[LANE:%.*]] = shufflevector <1 x double> [[NEG]]
  // CHECK: llvm.fma.v2f64(<2 x double> {{.*}}, <2 x double> [[LANE]], <2 x double> %a1)
  // CHECK-NEXT: ret
}
