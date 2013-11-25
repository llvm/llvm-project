// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>


float32_t test_vmuls_lane_f32(float32_t a, float32x2_t b) {
  // CHECK: test_vmuls_lane_f32
  return vmuls_lane_f32(a, b, 1);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float64_t test_vmuld_lane_f64(float64_t a, float64x1_t b) {
  // CHECK: test_vmuld_lane_f64
  return vmuld_lane_f64(a, b, 0);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float32_t test_vmuls_laneq_f32(float32_t a, float32x4_t b) {
  // CHECK: test_vmuls_laneq_f32
  return vmuls_laneq_f32(a, b, 3);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmuld_laneq_f64(float64_t a, float64x2_t b) {
  // CHECK: test_vmuld_laneq_f64
  return vmuld_laneq_f64(a, b, 1);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

float64x1_t test_vmul_n_f64(float64x1_t a, float64_t b) {
  // CHECK: test_vmul_n_f64
  return vmul_n_f64(a, b);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float32_t test_vmulxs_lane_f32(float32_t a, float32x2_t b) {
// CHECK: test_vmulxs_lane_f32
  return vmulxs_lane_f32(a, b, 1);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float32_t test_vmulxs_laneq_f32(float32_t a, float32x4_t b) {
// CHECK: test_vmulxs_laneq_f32
  return vmulxs_laneq_f32(a, b, 3);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmulxd_lane_f64(float64_t a, float64x1_t b) {
// CHECK: test_vmulxd_lane_f64
  return vmulxd_lane_f64(a, b, 0);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float64_t test_vmulxd_laneq_f64(float64_t a, float64x2_t b) {
// CHECK: test_vmulxd_laneq_f64
  return vmulxd_laneq_f64(a, b, 1);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vmulx_lane_f64
float64x1_t test_vmulx_lane_f64(float64x1_t a, float64x1_t b) {
  return vmulx_lane_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}


// CHECK: test_vmulx_laneq_f64_0
float64x1_t test_vmulx_laneq_f64_0(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vmulx_laneq_f64_1
float64x1_t test_vmulx_laneq_f64_1(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 1);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}


// CHECK: test_vfmas_lane_f32
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
  // CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK: test_vfmad_lane_f64
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vfmad_laneq_f64
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK: test_vfmss_lane_f32
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmss_lane_f32(a, b, c, 1);
  // CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK: test_vfma_lane_f64
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vfms_lane_f64
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfms_lane_f64(a, b, v, 0);
  // CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vfma_laneq_f64
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vfms_laneq_f64
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfms_laneq_f64(a, b, v, 0);
  // CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK: test_vqdmullh_lane_s16
int32_t test_vqdmullh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmullh_lane_s16(a, b, 3);
  // CHECK: sqdmull {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vqdmulls_lane_s32
int64_t test_vqdmulls_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulls_lane_s32(a, b, 1);
  // CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK: test_vqdmullh_laneq_s16
int32_t test_vqdmullh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmullh_laneq_s16(a, b, 7);
  // CHECK: sqdmull {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}

// CHECK: test_vqdmulls_laneq_s32
int64_t test_vqdmulls_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulls_laneq_s32(a, b, 3);
  // CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK: test_vqdmulhh_lane_s16
int16_t test_vqdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmulhh_lane_s16(a, b, 3);
// CHECK: sqdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vqdmulhs_lane_s32
int32_t test_vqdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulhs_lane_s32(a, b, 1);
// CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK: test_vqdmulhh_laneq_s16
int16_t test_vqdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmulhh_laneq_s16(a, b, 7);
// CHECK: sqdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}


// CHECK: test_vqdmulhs_laneq_s32
int32_t test_vqdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulhs_laneq_s32(a, b, 3);
// CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK: test_vqrdmulhh_lane_s16
int16_t test_vqrdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqrdmulhh_lane_s16(a, b, 3);
// CHECK: sqrdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vqrdmulhs_lane_s32
int32_t test_vqrdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqrdmulhs_lane_s32(a, b, 1);
// CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK: test_vqrdmulhh_laneq_s16
int16_t test_vqrdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqrdmulhh_laneq_s16(a, b, 7);
// CHECK: sqrdmulh {{h[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}


// CHECK: test_vqrdmulhs_laneq_s32
int32_t test_vqrdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqrdmulhs_laneq_s32(a, b, 3);
// CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK: test_vqdmlalh_lane_s16
int32_t test_vqdmlalh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlalh_lane_s16(a, b, c, 3);
// CHECK: sqdmlal {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vqdmlals_lane_s32
int64_t test_vqdmlals_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlals_lane_s32(a, b, c, 1);
// CHECK: sqdmlal {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK: test_vqdmlalh_laneq_s16
int32_t test_vqdmlalh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlalh_laneq_s16(a, b, c, 7);
// CHECK: sqdmlal {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}

// CHECK: test_vqdmlals_laneq_s32
int64_t test_vqdmlals_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlals_laneq_s32(a, b, c, 3);
// CHECK: sqdmlal {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK: test_vqdmlslh_lane_s16
int32_t test_vqdmlslh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlslh_lane_s16(a, b, c, 3);
// CHECK: sqdmlsl {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[3]
}

// CHECK: test_vqdmlsls_lane_s32
int64_t test_vqdmlsls_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlsls_lane_s32(a, b, c, 1);
// CHECK: sqdmlsl {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK: test_vqdmlslh_laneq_s16
int32_t test_vqdmlslh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlslh_laneq_s16(a, b, c, 7);
// CHECK: sqdmlsl {{s[0-9]+}}, {{h[0-9]+}}, {{v[0-9]+}}.h[7]
}

// CHECK: test_vqdmlsls_laneq_s32
int64_t test_vqdmlsls_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlsls_laneq_s32(a, b, c, 3);
// CHECK: sqdmlsl {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

