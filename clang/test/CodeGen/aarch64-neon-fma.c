// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

float32x2_t test_vmla_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  // CHECK-LABEL: test_vmla_n_f32
  return vmla_n_f32(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK-FMA: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float32x4_t test_vmlaq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  // CHECK-LABEL: test_vmlaq_n_f32
  return vmlaq_n_f32(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK-FMA: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float64x2_t test_vmlaq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  // CHECK-LABEL: test_vmlaq_n_f64
  return vmlaq_n_f64(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  // CHECK: fadd {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  // CHECK-FMA: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  // CHECK-FMA: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x4_t test_vmlsq_n_f32(float32x4_t a, float32x4_t b, float32_t c) {
  // CHECK-LABEL: test_vmlsq_n_f32
  return vmlsq_n_f32(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: dup {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK-FMA: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
}

float32x2_t test_vmls_n_f32(float32x2_t a, float32x2_t b, float32_t c) {
  // CHECK-LABEL: test_vmls_n_f32
  return vmls_n_f32(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: dup {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK-FMA: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
}

float64x2_t test_vmlsq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  // CHECK-LABEL: test_vmlsq_n_f64
  return vmlsq_n_f64(a, b, c);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  // CHECK: fsub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
  // CHECK-FMA: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
  // CHECK-FMA: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

float32x2_t test_vmla_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmla_lane_f32_0
  return vmla_lane_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmlaq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmlaq_lane_f32_0
  return vmlaq_lane_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmla_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmla_laneq_f32_0
  return vmla_laneq_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmlaq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmlaq_laneq_f32_0
  return vmlaq_laneq_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmls_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmls_lane_f32_0
  return vmls_lane_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmlsq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmlsq_lane_f32_0
  return vmlsq_lane_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmls_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmls_laneq_f32_0
  return vmls_laneq_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmlsq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmlsq_laneq_f32_0
  return vmlsq_laneq_f32(a, b, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmla_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmla_lane_f32
  return vmla_lane_f32(a, b, v, 1);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float32x4_t test_vmlaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmlaq_lane_f32
  return vmlaq_lane_f32(a, b, v, 1);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float32x2_t test_vmla_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmla_laneq_f32
  return vmla_laneq_f32(a, b, v, 3);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
  // CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float32x4_t test_vmlaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmlaq_laneq_f32
  return vmlaq_laneq_f32(a, b, v, 3);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
  // CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float32x2_t test_vmls_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmls_lane_f32
  return vmls_lane_f32(a, b, v, 1);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float32x4_t test_vmlsq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK-LABEL: test_vmlsq_lane_f32
  return vmlsq_lane_f32(a, b, v, 1);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}
float32x2_t test_vmls_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmls_laneq_f32
  return vmls_laneq_f32(a, b, v, 3);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
  // CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
  // CHECK-FMA: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float32x4_t test_vmlsq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK-LABEL: test_vmlsq_laneq_f32
  return vmlsq_laneq_f32(a, b, v, 3);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
  // CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
  // CHECK-FMA: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float64x2_t test_vfmaq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  // CHECK-LABEL: test_vfmaq_n_f64:
  return vfmaq_n_f64(a, b, c);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+\.2d|v[0-9]+\.d\[0\]}}
}

float64x2_t test_vfmsq_n_f64(float64x2_t a, float64x2_t b, float64_t c) {
  // CHECK-LABEL: test_vfmsq_n_f64:
  return vfmsq_n_f64(a, b, c);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+\.2d|v[0-9]+\.d\[0\]}}
}
