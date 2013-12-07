// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon \
// RUN:   -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>

int16x4_t test_vmla_lane_s16(int16x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmla_lane_s16
  return vmla_lane_s16(a, b, v, 3);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int16x8_t test_vmlaq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlaq_lane_s16
  return vmlaq_lane_s16(a, b, v, 3);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int32x2_t test_vmla_lane_s32(int32x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmla_lane_s32
  return vmla_lane_s32(a, b, v, 1);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlaq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlaq_lane_s32
  return vmlaq_lane_s32(a, b, v, 1);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int16x4_t test_vmla_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmla_laneq_s16
  return vmla_laneq_s16(a, b, v, 7);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int16x8_t test_vmlaq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlaq_laneq_s16
  return vmlaq_laneq_s16(a, b, v, 7);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int32x2_t test_vmla_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmla_laneq_s32
  return vmla_laneq_s32(a, b, v, 3);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlaq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlaq_laneq_s32
  return vmlaq_laneq_s32(a, b, v, 3);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int16x4_t test_vmls_lane_s16(int16x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmls_lane_s16
  return vmls_lane_s16(a, b, v, 3);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int16x8_t test_vmlsq_lane_s16(int16x8_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsq_lane_s16
  return vmlsq_lane_s16(a, b, v, 3);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int32x2_t test_vmls_lane_s32(int32x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmls_lane_s32
  return vmls_lane_s32(a, b, v, 1);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlsq_lane_s32(int32x4_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsq_lane_s32
  return vmlsq_lane_s32(a, b, v, 1);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int16x4_t test_vmls_laneq_s16(int16x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmls_laneq_s16
  return vmls_laneq_s16(a, b, v, 7);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int16x8_t test_vmlsq_laneq_s16(int16x8_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsq_laneq_s16
  return vmlsq_laneq_s16(a, b, v, 7);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int32x2_t test_vmls_laneq_s32(int32x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmls_laneq_s32
  return vmls_laneq_s32(a, b, v, 3);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlsq_laneq_s32(int32x4_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsq_laneq_s32
  return vmlsq_laneq_s32(a, b, v, 3);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int16x4_t test_vmul_lane_s16(int16x4_t a, int16x4_t v) {
  // CHECK: test_vmul_lane_s16
  return vmul_lane_s16(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int16x8_t test_vmulq_lane_s16(int16x8_t a, int16x4_t v) {
  // CHECK: test_vmulq_lane_s16
  return vmulq_lane_s16(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int32x2_t test_vmul_lane_s32(int32x2_t a, int32x2_t v) {
  // CHECK: test_vmul_lane_s32
  return vmul_lane_s32(a, v, 1);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmulq_lane_s32(int32x4_t a, int32x2_t v) {
  // CHECK: test_vmulq_lane_s32
  return vmulq_lane_s32(a, v, 1);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

uint16x4_t test_vmul_lane_u16(uint16x4_t a, uint16x4_t v) {
  // CHECK: test_vmul_lane_u16
  return vmul_lane_u16(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

uint16x8_t test_vmulq_lane_u16(uint16x8_t a, uint16x4_t v) {
  // CHECK: test_vmulq_lane_u16
  return vmulq_lane_u16(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

uint32x2_t test_vmul_lane_u32(uint32x2_t a, uint32x2_t v) {
  // CHECK: test_vmul_lane_u32
  return vmul_lane_u32(a, v, 1);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

uint32x4_t test_vmulq_lane_u32(uint32x4_t a, uint32x2_t v) {
  // CHECK: test_vmulq_lane_u32
  return vmulq_lane_u32(a, v, 1);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int16x4_t test_vmul_laneq_s16(int16x4_t a, int16x8_t v) {
  // CHECK: test_vmul_laneq_s16
  return vmul_laneq_s16(a, v, 7);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int16x8_t test_vmulq_laneq_s16(int16x8_t a, int16x8_t v) {
  // CHECK: test_vmulq_laneq_s16
  return vmulq_laneq_s16(a, v, 7);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int32x2_t test_vmul_laneq_s32(int32x2_t a, int32x4_t v) {
  // CHECK: test_vmul_laneq_s32
  return vmul_laneq_s32(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmulq_laneq_s32(int32x4_t a, int32x4_t v) {
  // CHECK: test_vmulq_laneq_s32
  return vmulq_laneq_s32(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

uint16x4_t test_vmul_laneq_u16(uint16x4_t a, uint16x8_t v) {
  // CHECK: test_vmul_laneq_u16
  return vmul_laneq_u16(a, v, 7);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

uint16x8_t test_vmulq_laneq_u16(uint16x8_t a, uint16x8_t v) {
  // CHECK: test_vmulq_laneq_u16
  return vmulq_laneq_u16(a, v, 7);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

uint32x2_t test_vmul_laneq_u32(uint32x2_t a, uint32x4_t v) {
  // CHECK: test_vmul_laneq_u32
  return vmul_laneq_u32(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

uint32x4_t test_vmulq_laneq_u32(uint32x4_t a, uint32x4_t v) {
  // CHECK: test_vmulq_laneq_u32
  return vmulq_laneq_u32(a, v, 3);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float32x2_t test_vfma_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK: test_vfma_lane_f32
  return vfma_lane_f32(a, b, v, 1);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float32x4_t test_vfmaq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK: test_vfmaq_lane_f32
  return vfmaq_lane_f32(a, b, v, 1);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float32x2_t test_vfma_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK: test_vfma_laneq_f32
  return vfma_laneq_f32(a, b, v, 3);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float32x4_t test_vfmaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK: test_vfmaq_laneq_f32
  return vfmaq_laneq_f32(a, b, v, 3);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float32x2_t test_vfms_lane_f32(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK: test_vfms_lane_f32
  return vfms_lane_f32(a, b, v, 1);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float32x4_t test_vfmsq_lane_f32(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK: test_vfmsq_lane_f32
  return vfmsq_lane_f32(a, b, v, 1);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float32x2_t test_vfms_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK: test_vfms_laneq_f32
  return vfms_laneq_f32(a, b, v, 3);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float32x4_t test_vfmsq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK: test_vfmsq_laneq_f32
  return vfmsq_laneq_f32(a, b, v, 3);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float64x2_t test_vfmaq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  // CHECK: test_vfmaq_lane_f64
  return vfmaq_lane_f64(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float64x2_t test_vfmaq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  // CHECK: test_vfmaq_laneq_f64
  return vfmaq_laneq_f64(a, b, v, 1);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
}

float64x2_t test_vfmsq_lane_f64(float64x2_t a, float64x2_t b, float64x1_t v) {
  // CHECK: test_vfmsq_lane_f64
  return vfmsq_lane_f64(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float64x2_t test_vfmsq_laneq_f64(float64x2_t a, float64x2_t b, float64x2_t v) {
  // CHECK: test_vfmsq_laneq_f64
  return vfmsq_laneq_f64(a, b, v, 1);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
}

int32x4_t test_vmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlal_lane_s16
  return vmlal_lane_s16(a, b, v, 3);
  // CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlal_lane_s32
  return vmlal_lane_s32(a, b, v, 1);
  // CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlal_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlal_laneq_s16
  return vmlal_laneq_s16(a, b, v, 7);
  // CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlal_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlal_laneq_s32
  return vmlal_laneq_s32(a, b, v, 3);
  // CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlal_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlal_high_lane_s16
  return vmlal_high_lane_s16(a, b, v, 3);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlal_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlal_high_lane_s32
  return vmlal_high_lane_s32(a, b, v, 1);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlal_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlal_high_laneq_s16
  return vmlal_high_laneq_s16(a, b, v, 7);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlal_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlal_high_laneq_s32
  return vmlal_high_laneq_s32(a, b, v, 3);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlsl_lane_s16
  return vmlsl_lane_s16(a, b, v, 3);
  // CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlsl_lane_s32
  return vmlsl_lane_s32(a, b, v, 1);
  // CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlsl_laneq_s16(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlsl_laneq_s16
  return vmlsl_laneq_s16(a, b, v, 7);
  // CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlsl_laneq_s32(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlsl_laneq_s32
  return vmlsl_laneq_s32(a, b, v, 3);
  // CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlsl_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsl_high_lane_s16
  return vmlsl_high_lane_s16(a, b, v, 3);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlsl_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsl_high_lane_s32
  return vmlsl_high_lane_s32(a, b, v, 1);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlsl_high_laneq_s16(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsl_high_laneq_s16
  return vmlsl_high_laneq_s16(a, b, v, 7);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlsl_high_laneq_s32(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsl_high_laneq_s32
  return vmlsl_high_laneq_s32(a, b, v, 3);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlal_lane_u16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlal_lane_u16
  return vmlal_lane_u16(a, b, v, 3);
  // CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlal_lane_u32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlal_lane_u32
  return vmlal_lane_u32(a, b, v, 1);
  // CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlal_laneq_u16(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlal_laneq_u16
  return vmlal_laneq_u16(a, b, v, 7);
  // CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlal_laneq_u32(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlal_laneq_u32
  return vmlal_laneq_u32(a, b, v, 3);
  // CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlal_high_lane_u16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlal_high_lane_u16
  return vmlal_high_lane_u16(a, b, v, 3);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlal_high_lane_u32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlal_high_lane_u32
  return vmlal_high_lane_u32(a, b, v, 1);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlal_high_laneq_u16(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlal_high_laneq_u16
  return vmlal_high_laneq_u16(a, b, v, 7);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlal_high_laneq_u32(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlal_high_laneq_u32
  return vmlal_high_laneq_u32(a, b, v, 3);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlsl_lane_u16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlsl_lane_u16
  return vmlsl_lane_u16(a, b, v, 3);
  // CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlsl_lane_u32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlsl_lane_u32
  return vmlsl_lane_u32(a, b, v, 1);
  // CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlsl_laneq_u16(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlsl_laneq_u16
  return vmlsl_laneq_u16(a, b, v, 7);
  // CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlsl_laneq_u32(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlsl_laneq_u32
  return vmlsl_laneq_u32(a, b, v, 3);
  // CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmlsl_high_lane_u16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsl_high_lane_u16
  return vmlsl_high_lane_u16(a, b, v, 3);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmlsl_high_lane_u32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsl_high_lane_u32
  return vmlsl_high_lane_u32(a, b, v, 1);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmlsl_high_laneq_u16(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsl_high_laneq_u16
  return vmlsl_high_laneq_u16(a, b, v, 7);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmlsl_high_laneq_u32(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsl_high_laneq_u32
  return vmlsl_high_laneq_u32(a, b, v, 3);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmull_lane_s16(int16x4_t a, int16x4_t v) {
  // CHECK: test_vmull_lane_s16
  return vmull_lane_s16(a, v, 3);
  // CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmull_lane_s32(int32x2_t a, int32x2_t v) {
  // CHECK: test_vmull_lane_s32
  return vmull_lane_s32(a, v, 1);
  // CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

uint32x4_t test_vmull_lane_u16(uint16x4_t a, uint16x4_t v) {
  // CHECK: test_vmull_lane_u16
  return vmull_lane_u16(a, v, 3);
  // CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

uint64x2_t test_vmull_lane_u32(uint32x2_t a, uint32x2_t v) {
  // CHECK: test_vmull_lane_u32
  return vmull_lane_u32(a, v, 1);
  // CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmull_high_lane_s16(int16x8_t a, int16x4_t v) {
  // CHECK: test_vmull_high_lane_s16
  return vmull_high_lane_s16(a, v, 3);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vmull_high_lane_s32(int32x4_t a, int32x2_t v) {
  // CHECK: test_vmull_high_lane_s32
  return vmull_high_lane_s32(a, v, 1);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

uint32x4_t test_vmull_high_lane_u16(uint16x8_t a, uint16x4_t v) {
  // CHECK: test_vmull_high_lane_u16
  return vmull_high_lane_u16(a, v, 3);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

uint64x2_t test_vmull_high_lane_u32(uint32x4_t a, uint32x2_t v) {
  // CHECK: test_vmull_high_lane_u32
  return vmull_high_lane_u32(a, v, 1);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vmull_laneq_s16(int16x4_t a, int16x8_t v) {
  // CHECK: test_vmull_laneq_s16
  return vmull_laneq_s16(a, v, 7);
  // CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmull_laneq_s32(int32x2_t a, int32x4_t v) {
  // CHECK: test_vmull_laneq_s32
  return vmull_laneq_s32(a, v, 3);
  // CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

uint32x4_t test_vmull_laneq_u16(uint16x4_t a, uint16x8_t v) {
  // CHECK: test_vmull_laneq_u16
  return vmull_laneq_u16(a, v, 7);
  // CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[7]
}

uint64x2_t test_vmull_laneq_u32(uint32x2_t a, uint32x4_t v) {
  // CHECK: test_vmull_laneq_u32
  return vmull_laneq_u32(a, v, 3);
  // CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vmull_high_laneq_s16(int16x8_t a, int16x8_t v) {
  // CHECK: test_vmull_high_laneq_s16
  return vmull_high_laneq_s16(a, v, 7);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vmull_high_laneq_s32(int32x4_t a, int32x4_t v) {
  // CHECK: test_vmull_high_laneq_s32
  return vmull_high_laneq_s32(a, v, 3);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

uint32x4_t test_vmull_high_laneq_u16(uint16x8_t a, uint16x8_t v) {
  // CHECK: test_vmull_high_laneq_u16
  return vmull_high_laneq_u16(a, v, 7);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

uint64x2_t test_vmull_high_laneq_u32(uint32x4_t a, uint32x4_t v) {
  // CHECK: test_vmull_high_laneq_u32
  return vmull_high_laneq_u32(a, v, 3);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vqdmlal_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vqdmlal_lane_s16
  return vqdmlal_lane_s16(a, b, v, 3);
  // CHECK: sqdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmlal_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vqdmlal_lane_s32
  return vqdmlal_lane_s32(a, b, v, 1);
  // CHECK: sqdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmlal_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vqdmlal_high_lane_s16
  return vqdmlal_high_lane_s16(a, b, v, 3);
  // CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmlal_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vqdmlal_high_lane_s32
  return vqdmlal_high_lane_s32(a, b, v, 1);
  // CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmlsl_lane_s16(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vqdmlsl_lane_s16
  return vqdmlsl_lane_s16(a, b, v, 3);
  // CHECK: sqdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmlsl_lane_s32(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vqdmlsl_lane_s32
  return vqdmlsl_lane_s32(a, b, v, 1);
  // CHECK: sqdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmlsl_high_lane_s16(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vqdmlsl_high_lane_s16
  return vqdmlsl_high_lane_s16(a, b, v, 3);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmlsl_high_lane_s32(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vqdmlsl_high_lane_s32
  return vqdmlsl_high_lane_s32(a, b, v, 1);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmull_lane_s16(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqdmull_lane_s16
  return vqdmull_lane_s16(a, v, 3);
  // CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmull_lane_s32(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqdmull_lane_s32
  return vqdmull_lane_s32(a, v, 1);
  // CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmull_laneq_s16(int16x4_t a, int16x8_t v) {
  // CHECK: test_vqdmull_laneq_s16
  return vqdmull_laneq_s16(a, v, 3);
  // CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmull_laneq_s32(int32x2_t a, int32x4_t v) {
  // CHECK: test_vqdmull_laneq_s32
  return vqdmull_laneq_s32(a, v, 3);
  // CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

int32x4_t test_vqdmull_high_lane_s16(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqdmull_high_lane_s16
  return vqdmull_high_lane_s16(a, v, 3);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int64x2_t test_vqdmull_high_lane_s32(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqdmull_high_lane_s32
  return vqdmull_high_lane_s32(a, v, 1);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmull_high_laneq_s16(int16x8_t a, int16x8_t v) {
  // CHECK: test_vqdmull_high_laneq_s16
  return vqdmull_high_laneq_s16(a, v, 7);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[7]
}

int64x2_t test_vqdmull_high_laneq_s32(int32x4_t a, int32x4_t v) {
  // CHECK: test_vqdmull_high_laneq_s32
  return vqdmull_high_laneq_s32(a, v, 3);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

int16x4_t test_vqdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqdmulh_lane_s16
  return vqdmulh_lane_s16(a, v, 3);
  // CHECK: sqdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int16x8_t test_vqdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqdmulhq_lane_s16
  return vqdmulhq_lane_s16(a, v, 3);
  // CHECK: sqdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int32x2_t test_vqdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqdmulh_lane_s32
  return vqdmulh_lane_s32(a, v, 1);
  // CHECK: sqdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqdmulhq_lane_s32
  return vqdmulhq_lane_s32(a, v, 1);
  // CHECK: sqdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

int16x4_t test_vqrdmulh_lane_s16(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqrdmulh_lane_s16
  return vqrdmulh_lane_s16(a, v, 3);
  // CHECK: sqrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[3]
}

int16x8_t test_vqrdmulhq_lane_s16(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqrdmulhq_lane_s16
  return vqrdmulhq_lane_s16(a, v, 3);
  // CHECK: sqrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[3]
}

int32x2_t test_vqrdmulh_lane_s32(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqrdmulh_lane_s32
  return vqrdmulh_lane_s32(a, v, 1);
  // CHECK: sqrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

int32x4_t test_vqrdmulhq_lane_s32(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqrdmulhq_lane_s32
  return vqrdmulhq_lane_s32(a, v, 1);
  // CHECK: sqrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float32x2_t test_vmul_lane_f32(float32x2_t a, float32x2_t v) {
  // CHECK: test_vmul_lane_f32
  return vmul_lane_f32(a, v, 1);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}


float64x1_t test_vmul_lane_f64(float64x1_t a, float64x1_t v) {
  // CHECK: test_vmul_lane_f64
  return vmul_lane_f64(a, v, 0);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}


float32x4_t test_vmulq_lane_f32(float32x4_t a, float32x2_t v) {
  // CHECK: test_vmulq_lane_f32
  return vmulq_lane_f32(a, v, 1);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float64x2_t test_vmulq_lane_f64(float64x2_t a, float64x1_t v) {
  // CHECK: test_vmulq_lane_f64
  return vmulq_lane_f64(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float32x2_t test_vmul_laneq_f32(float32x2_t a, float32x4_t v) {
  // CHECK: test_vmul_laneq_f32
  return vmul_laneq_f32(a, v, 3);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float64x1_t test_vmul_laneq_f64(float64x1_t a, float64x2_t v) {
  // CHECK: test_vmul_laneq_f64
  return vmul_laneq_f64(a, v, 1);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}


float32x4_t test_vmulq_laneq_f32(float32x4_t a, float32x4_t v) {
  // CHECK: test_vmulq_laneq_f32
  return vmulq_laneq_f32(a, v, 3);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float64x2_t test_vmulq_laneq_f64(float64x2_t a, float64x2_t v) {
  // CHECK: test_vmulq_laneq_f64
  return vmulq_laneq_f64(a, v, 1);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
}

float32x2_t test_vmulx_lane_f32(float32x2_t a, float32x2_t v) {
  // CHECK: test_vmulx_lane_f32
  return vmulx_lane_f32(a, v, 1);
  // CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[1]
}

float32x4_t test_vmulxq_lane_f32(float32x4_t a, float32x2_t v) {
  // CHECK: test_vmulxq_lane_f32
  return vmulxq_lane_f32(a, v, 1);
  // CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[1]
}

float64x2_t test_vmulxq_lane_f64(float64x2_t a, float64x1_t v) {
  // CHECK: test_vmulxq_lane_f64
  return vmulxq_lane_f64(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float32x2_t test_vmulx_laneq_f32(float32x2_t a, float32x4_t v) {
  // CHECK: test_vmulx_laneq_f32
  return vmulx_laneq_f32(a, v, 3);
  // CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[3]
}

float32x4_t test_vmulxq_laneq_f32(float32x4_t a, float32x4_t v) {
  // CHECK: test_vmulxq_laneq_f32
  return vmulxq_laneq_f32(a, v, 3);
  // CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[3]
}

float64x2_t test_vmulxq_laneq_f64(float64x2_t a, float64x2_t v) {
  // CHECK: test_vmulxq_laneq_f64
  return vmulxq_laneq_f64(a, v, 1);
  // CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
}

int16x4_t test_vmla_lane_s16_0(int16x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmla_lane_s16_0
  return vmla_lane_s16(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmlaq_lane_s16_0(int16x8_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlaq_lane_s16_0
  return vmlaq_lane_s16(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmla_lane_s32_0(int32x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmla_lane_s32_0
  return vmla_lane_s32(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlaq_lane_s32_0(int32x4_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlaq_lane_s32_0
  return vmlaq_lane_s32(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vmla_laneq_s16_0(int16x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmla_laneq_s16_0
  return vmla_laneq_s16(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmlaq_laneq_s16_0(int16x8_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlaq_laneq_s16_0
  return vmlaq_laneq_s16(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmla_laneq_s32_0(int32x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmla_laneq_s32_0
  return vmla_laneq_s32(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlaq_laneq_s32_0(int32x4_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlaq_laneq_s32_0
  return vmlaq_laneq_s32(a, b, v, 0);
  // CHECK: mla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vmls_lane_s16_0(int16x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmls_lane_s16_0
  return vmls_lane_s16(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmlsq_lane_s16_0(int16x8_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsq_lane_s16_0
  return vmlsq_lane_s16(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmls_lane_s32_0(int32x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmls_lane_s32_0
  return vmls_lane_s32(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsq_lane_s32_0(int32x4_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsq_lane_s32_0
  return vmlsq_lane_s32(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vmls_laneq_s16_0(int16x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmls_laneq_s16_0
  return vmls_laneq_s16(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmlsq_laneq_s16_0(int16x8_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsq_laneq_s16_0
  return vmlsq_laneq_s16(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmls_laneq_s32_0(int32x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmls_laneq_s32_0
  return vmls_laneq_s32(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsq_laneq_s32_0(int32x4_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsq_laneq_s32_0
  return vmlsq_laneq_s32(a, b, v, 0);
  // CHECK: mls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vmul_lane_s16_0(int16x4_t a, int16x4_t v) {
  // CHECK: test_vmul_lane_s16_0
  return vmul_lane_s16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmulq_lane_s16_0(int16x8_t a, int16x4_t v) {
  // CHECK: test_vmulq_lane_s16_0
  return vmulq_lane_s16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmul_lane_s32_0(int32x2_t a, int32x2_t v) {
  // CHECK: test_vmul_lane_s32_0
  return vmul_lane_s32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmulq_lane_s32_0(int32x4_t a, int32x2_t v) {
  // CHECK: test_vmulq_lane_s32_0
  return vmulq_lane_s32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint16x4_t test_vmul_lane_u16_0(uint16x4_t a, uint16x4_t v) {
  // CHECK: test_vmul_lane_u16_0
  return vmul_lane_u16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

uint16x8_t test_vmulq_lane_u16_0(uint16x8_t a, uint16x4_t v) {
  // CHECK: test_vmulq_lane_u16_0
  return vmulq_lane_u16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint32x2_t test_vmul_lane_u32_0(uint32x2_t a, uint32x2_t v) {
  // CHECK: test_vmul_lane_u32_0
  return vmul_lane_u32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmulq_lane_u32_0(uint32x4_t a, uint32x2_t v) {
  // CHECK: test_vmulq_lane_u32_0
  return vmulq_lane_u32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vmul_laneq_s16_0(int16x4_t a, int16x8_t v) {
  // CHECK: test_vmul_laneq_s16_0
  return vmul_laneq_s16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vmulq_laneq_s16_0(int16x8_t a, int16x8_t v) {
  // CHECK: test_vmulq_laneq_s16_0
  return vmulq_laneq_s16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vmul_laneq_s32_0(int32x2_t a, int32x4_t v) {
  // CHECK: test_vmul_laneq_s32_0
  return vmul_laneq_s32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmulq_laneq_s32_0(int32x4_t a, int32x4_t v) {
  // CHECK: test_vmulq_laneq_s32_0
  return vmulq_laneq_s32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint16x4_t test_vmul_laneq_u16_0(uint16x4_t a, uint16x8_t v) {
  // CHECK: test_vmul_laneq_u16_0
  return vmul_laneq_u16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

uint16x8_t test_vmulq_laneq_u16_0(uint16x8_t a, uint16x8_t v) {
  // CHECK: test_vmulq_laneq_u16_0
  return vmulq_laneq_u16(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint32x2_t test_vmul_laneq_u32_0(uint32x2_t a, uint32x4_t v) {
  // CHECK: test_vmul_laneq_u32_0
  return vmul_laneq_u32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmulq_laneq_u32_0(uint32x4_t a, uint32x4_t v) {
  // CHECK: test_vmulq_laneq_u32_0
  return vmulq_laneq_u32(a, v, 0);
  // CHECK: mul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vfma_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK: test_vfma_lane_f32_0
  return vfma_lane_f32(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmaq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK: test_vfmaq_lane_f32_0
  return vfmaq_lane_f32(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vfma_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK: test_vfma_laneq_f32_0
  return vfma_laneq_f32(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmaq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK: test_vfmaq_laneq_f32_0
  return vfmaq_laneq_f32(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vfms_lane_f32_0(float32x2_t a, float32x2_t b, float32x2_t v) {
  // CHECK: test_vfms_lane_f32_0
  return vfms_lane_f32(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmsq_lane_f32_0(float32x4_t a, float32x4_t b, float32x2_t v) {
  // CHECK: test_vfmsq_lane_f32_0
  return vfmsq_lane_f32(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vfms_laneq_f32_0(float32x2_t a, float32x2_t b, float32x4_t v) {
  // CHECK: test_vfms_laneq_f32_0
  return vfms_laneq_f32(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmsq_laneq_f32_0(float32x4_t a, float32x4_t b, float32x4_t v) {
  // CHECK: test_vfmsq_laneq_f32_0
  return vfmsq_laneq_f32(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vfmaq_laneq_f64_0(float64x2_t a, float64x2_t b, float64x2_t v) {
  // CHECK: test_vfmaq_laneq_f64_0
  return vfmaq_laneq_f64(a, b, v, 0);
  // CHECK: fmla {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float64x2_t test_vfmsq_laneq_f64_0(float64x2_t a, float64x2_t b, float64x2_t v) {
  // CHECK: test_vfmsq_laneq_f64_0
  return vfmsq_laneq_f64(a, b, v, 0);
  // CHECK: fmls {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

int32x4_t test_vmlal_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlal_lane_s16_0
  return vmlal_lane_s16(a, b, v, 0);
  // CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlal_lane_s32_0
  return vmlal_lane_s32(a, b, v, 0);
  // CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlal_laneq_s16_0
  return vmlal_laneq_s16(a, b, v, 0);
  // CHECK: smlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlal_laneq_s32_0
  return vmlal_laneq_s32(a, b, v, 0);
  // CHECK: smlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlal_high_lane_s16_0
  return vmlal_high_lane_s16(a, b, v, 0);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlal_high_lane_s32_0
  return vmlal_high_lane_s32(a, b, v, 0);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlal_high_laneq_s16_0
  return vmlal_high_laneq_s16(a, b, v, 0);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlal_high_laneq_s32_0
  return vmlal_high_laneq_s32(a, b, v, 0);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlsl_lane_s16_0
  return vmlsl_lane_s16(a, b, v, 0);
  // CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlsl_lane_s32_0
  return vmlsl_lane_s32(a, b, v, 0);
  // CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_laneq_s16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlsl_laneq_s16_0
  return vmlsl_laneq_s16(a, b, v, 0);
  // CHECK: smlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_laneq_s32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlsl_laneq_s32_0
  return vmlsl_laneq_s32(a, b, v, 0);
  // CHECK: smlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsl_high_lane_s16_0
  return vmlsl_high_lane_s16(a, b, v, 0);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsl_high_lane_s32_0
  return vmlsl_high_lane_s32(a, b, v, 0);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_high_laneq_s16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsl_high_laneq_s16_0
  return vmlsl_high_laneq_s16(a, b, v, 0);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_high_laneq_s32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsl_high_laneq_s32_0
  return vmlsl_high_laneq_s32(a, b, v, 0);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_lane_u16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlal_lane_u16_0
  return vmlal_lane_u16(a, b, v, 0);
  // CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_lane_u32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlal_lane_u32_0
  return vmlal_lane_u32(a, b, v, 0);
  // CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_laneq_u16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlal_laneq_u16_0
  return vmlal_laneq_u16(a, b, v, 0);
  // CHECK: umlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_laneq_u32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlal_laneq_u32_0
  return vmlal_laneq_u32(a, b, v, 0);
  // CHECK: umlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_high_lane_u16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlal_high_lane_u16_0
  return vmlal_high_lane_u16(a, b, v, 0);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_high_lane_u32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlal_high_lane_u32_0
  return vmlal_high_lane_u32(a, b, v, 0);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_high_laneq_u16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlal_high_laneq_u16_0
  return vmlal_high_laneq_u16(a, b, v, 0);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_high_laneq_u32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlal_high_laneq_u32_0
  return vmlal_high_laneq_u32(a, b, v, 0);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_lane_u16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vmlsl_lane_u16_0
  return vmlsl_lane_u16(a, b, v, 0);
  // CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_lane_u32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vmlsl_lane_u32_0
  return vmlsl_lane_u32(a, b, v, 0);
  // CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_laneq_u16_0(int32x4_t a, int16x4_t b, int16x8_t v) {
  // CHECK: test_vmlsl_laneq_u16_0
  return vmlsl_laneq_u16(a, b, v, 0);
  // CHECK: umlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_laneq_u32_0(int64x2_t a, int32x2_t b, int32x4_t v) {
  // CHECK: test_vmlsl_laneq_u32_0
  return vmlsl_laneq_u32(a, b, v, 0);
  // CHECK: umlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_high_lane_u16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vmlsl_high_lane_u16_0
  return vmlsl_high_lane_u16(a, b, v, 0);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_high_lane_u32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vmlsl_high_lane_u32_0
  return vmlsl_high_lane_u32(a, b, v, 0);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_high_laneq_u16_0(int32x4_t a, int16x8_t b, int16x8_t v) {
  // CHECK: test_vmlsl_high_laneq_u16_0
  return vmlsl_high_laneq_u16(a, b, v, 0);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_high_laneq_u32_0(int64x2_t a, int32x4_t b, int32x4_t v) {
  // CHECK: test_vmlsl_high_laneq_u32_0
  return vmlsl_high_laneq_u32(a, b, v, 0);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmull_lane_s16_0(int16x4_t a, int16x4_t v) {
  // CHECK: test_vmull_lane_s16_0
  return vmull_lane_s16(a, v, 0);
  // CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmull_lane_s32_0(int32x2_t a, int32x2_t v) {
  // CHECK: test_vmull_lane_s32_0
  return vmull_lane_s32(a, v, 0);
  // CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmull_lane_u16_0(uint16x4_t a, uint16x4_t v) {
  // CHECK: test_vmull_lane_u16_0
  return vmull_lane_u16(a, v, 0);
  // CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmull_lane_u32_0(uint32x2_t a, uint32x2_t v) {
  // CHECK: test_vmull_lane_u32_0
  return vmull_lane_u32(a, v, 0);
  // CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmull_high_lane_s16_0(int16x8_t a, int16x4_t v) {
  // CHECK: test_vmull_high_lane_s16_0
  return vmull_high_lane_s16(a, v, 0);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmull_high_lane_s32_0(int32x4_t a, int32x2_t v) {
  // CHECK: test_vmull_high_lane_s32_0
  return vmull_high_lane_s32(a, v, 0);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmull_high_lane_u16_0(uint16x8_t a, uint16x4_t v) {
  // CHECK: test_vmull_high_lane_u16_0
  return vmull_high_lane_u16(a, v, 0);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmull_high_lane_u32_0(uint32x4_t a, uint32x2_t v) {
  // CHECK: test_vmull_high_lane_u32_0
  return vmull_high_lane_u32(a, v, 0);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmull_laneq_s16_0(int16x4_t a, int16x8_t v) {
  // CHECK: test_vmull_laneq_s16_0
  return vmull_laneq_s16(a, v, 0);
  // CHECK: smull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmull_laneq_s32_0(int32x2_t a, int32x4_t v) {
  // CHECK: test_vmull_laneq_s32_0
  return vmull_laneq_s32(a, v, 0);
  // CHECK: smull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmull_laneq_u16_0(uint16x4_t a, uint16x8_t v) {
  // CHECK: test_vmull_laneq_u16_0
  return vmull_laneq_u16(a, v, 0);
  // CHECK: umull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmull_laneq_u32_0(uint32x2_t a, uint32x4_t v) {
  // CHECK: test_vmull_laneq_u32_0
  return vmull_laneq_u32(a, v, 0);
  // CHECK: umull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmull_high_laneq_s16_0(int16x8_t a, int16x8_t v) {
  // CHECK: test_vmull_high_laneq_s16_0
  return vmull_high_laneq_s16(a, v, 0);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmull_high_laneq_s32_0(int32x4_t a, int32x4_t v) {
  // CHECK: test_vmull_high_laneq_s32_0
  return vmull_high_laneq_s32(a, v, 0);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmull_high_laneq_u16_0(uint16x8_t a, uint16x8_t v) {
  // CHECK: test_vmull_high_laneq_u16_0
  return vmull_high_laneq_u16(a, v, 0);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmull_high_laneq_u32_0(uint32x4_t a, uint32x4_t v) {
  // CHECK: test_vmull_high_laneq_u32_0
  return vmull_high_laneq_u32(a, v, 0);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlal_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vqdmlal_lane_s16_0
  return vqdmlal_lane_s16(a, b, v, 0);
  // CHECK: sqdmlal {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlal_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vqdmlal_lane_s32_0
  return vqdmlal_lane_s32(a, b, v, 0);
  // CHECK: sqdmlal {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlal_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vqdmlal_high_lane_s16_0
  return vqdmlal_high_lane_s16(a, b, v, 0);
  // CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlal_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vqdmlal_high_lane_s32_0
  return vqdmlal_high_lane_s32(a, b, v, 0);
  // CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlsl_lane_s16_0(int32x4_t a, int16x4_t b, int16x4_t v) {
  // CHECK: test_vqdmlsl_lane_s16_0
  return vqdmlsl_lane_s16(a, b, v, 0);
  // CHECK: sqdmlsl {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlsl_lane_s32_0(int64x2_t a, int32x2_t b, int32x2_t v) {
  // CHECK: test_vqdmlsl_lane_s32_0
  return vqdmlsl_lane_s32(a, b, v, 0);
  // CHECK: sqdmlsl {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlsl_high_lane_s16_0(int32x4_t a, int16x8_t b, int16x4_t v) {
  // CHECK: test_vqdmlsl_high_lane_s16_0
  return vqdmlsl_high_lane_s16(a, b, v, 0);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlsl_high_lane_s32_0(int64x2_t a, int32x4_t b, int32x2_t v) {
  // CHECK: test_vqdmlsl_high_lane_s32_0
  return vqdmlsl_high_lane_s32(a, b, v, 0);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmull_lane_s16_0(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqdmull_lane_s16_0
  return vqdmull_lane_s16(a, v, 0);
  // CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmull_lane_s32_0(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqdmull_lane_s32_0
  return vqdmull_lane_s32(a, v, 0);
  // CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmull_laneq_s16_0(int16x4_t a, int16x8_t v) {
  // CHECK: test_vqdmull_laneq_s16_0
  return vqdmull_laneq_s16(a, v, 0);
  // CHECK: sqdmull {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmull_laneq_s32_0(int32x2_t a, int32x4_t v) {
  // CHECK: test_vqdmull_laneq_s32_0
  return vqdmull_laneq_s32(a, v, 0);
  // CHECK: sqdmull {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmull_high_lane_s16_0(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqdmull_high_lane_s16_0
  return vqdmull_high_lane_s16(a, v, 0);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmull_high_lane_s32_0(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqdmull_high_lane_s32_0
  return vqdmull_high_lane_s32(a, v, 0);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmull_high_laneq_s16_0(int16x8_t a, int16x8_t v) {
  // CHECK: test_vqdmull_high_laneq_s16_0
  return vqdmull_high_laneq_s16(a, v, 0);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmull_high_laneq_s32_0(int32x4_t a, int32x4_t v) {
  // CHECK: test_vqdmull_high_laneq_s32_0
  return vqdmull_high_laneq_s32(a, v, 0);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vqdmulh_lane_s16_0(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqdmulh_lane_s16_0
  return vqdmulh_lane_s16(a, v, 0);
  // CHECK: sqdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vqdmulhq_lane_s16_0(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqdmulhq_lane_s16_0
  return vqdmulhq_lane_s16(a, v, 0);
  // CHECK: sqdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vqdmulh_lane_s32_0(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqdmulh_lane_s32_0
  return vqdmulh_lane_s32(a, v, 0);
  // CHECK: sqdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmulhq_lane_s32_0(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqdmulhq_lane_s32_0
  return vqdmulhq_lane_s32(a, v, 0);
  // CHECK: sqdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int16x4_t test_vqrdmulh_lane_s16_0(int16x4_t a, int16x4_t v) {
  // CHECK: test_vqrdmulh_lane_s16_0
  return vqrdmulh_lane_s16(a, v, 0);
  // CHECK: sqrdmulh {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.h[0]
}

int16x8_t test_vqrdmulhq_lane_s16_0(int16x8_t a, int16x4_t v) {
  // CHECK: test_vqrdmulhq_lane_s16_0
  return vqrdmulhq_lane_s16(a, v, 0);
  // CHECK: sqrdmulh {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int32x2_t test_vqrdmulh_lane_s32_0(int32x2_t a, int32x2_t v) {
  // CHECK: test_vqrdmulh_lane_s32_0
  return vqrdmulh_lane_s32(a, v, 0);
  // CHECK: sqrdmulh {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqrdmulhq_lane_s32_0(int32x4_t a, int32x2_t v) {
  // CHECK: test_vqrdmulhq_lane_s32_0
  return vqrdmulhq_lane_s32(a, v, 0);
  // CHECK: sqrdmulh {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmul_lane_f32_0(float32x2_t a, float32x2_t v) {
  // CHECK: test_vmul_lane_f32_0
  return vmul_lane_f32(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmulq_lane_f32_0(float32x4_t a, float32x2_t v) {
  // CHECK: test_vmulq_lane_f32_0
  return vmulq_lane_f32(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmul_laneq_f32_0(float32x2_t a, float32x4_t v) {
  // CHECK: test_vmul_laneq_f32_0
  return vmul_laneq_f32(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float64x1_t test_vmul_laneq_f64_0(float64x1_t a, float64x2_t v) {
  // CHECK: test_vmul_laneq_f64_0
  return vmul_laneq_f64(a, v, 0);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

float32x4_t test_vmulq_laneq_f32_0(float32x4_t a, float32x4_t v) {
  // CHECK: test_vmulq_laneq_f32_0
  return vmulq_laneq_f32(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vmulq_laneq_f64_0(float64x2_t a, float64x2_t v) {
  // CHECK: test_vmulq_laneq_f64_0
  return vmulq_laneq_f64(a, v, 0);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float32x2_t test_vmulx_lane_f32_0(float32x2_t a, float32x2_t v) {
  // CHECK: test_vmulx_lane_f32_0
  return vmulx_lane_f32(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmulxq_lane_f32_0(float32x4_t a, float32x2_t v) {
  // CHECK: test_vmulxq_lane_f32_0
  return vmulxq_lane_f32(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vmulxq_lane_f64_0(float64x2_t a, float64x1_t v) {
  // CHECK: test_vmulxq_lane_f64_0
  return vmulxq_lane_f64(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float32x2_t test_vmulx_laneq_f32_0(float32x2_t a, float32x4_t v) {
  // CHECK: test_vmulx_laneq_f32_0
  return vmulx_laneq_f32(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmulxq_laneq_f32_0(float32x4_t a, float32x4_t v) {
  // CHECK: test_vmulxq_laneq_f32_0
  return vmulxq_laneq_f32(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vmulxq_laneq_f64_0(float64x2_t a, float64x2_t v) {
  // CHECK: test_vmulxq_laneq_f64_0
  return vmulxq_laneq_f64(a, v, 0);
  // CHECK: fmulx {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

int32x4_t test_vmull_high_n_s16(int16x8_t a, int16_t b) {
  // CHECK: test_vmull_high_n_s16
  return vmull_high_n_s16(a, b);
  // CHECK: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmull_high_n_s32(int32x4_t a, int32_t b) {
  // CHECK: test_vmull_high_n_s32
  return vmull_high_n_s32(a, b);
  // CHECK: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmull_high_n_u16(uint16x8_t a, uint16_t b) {
  // CHECK: test_vmull_high_n_u16
  return vmull_high_n_u16(a, b);
  // CHECK: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmull_high_n_u32(uint32x4_t a, uint32_t b) {
  // CHECK: test_vmull_high_n_u32
  return vmull_high_n_u32(a, b);
  // CHECK: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmull_high_n_s16(int16x8_t a, int16_t b) {
  // CHECK: test_vqdmull_high_n_s16
  return vqdmull_high_n_s16(a, b);
  // CHECK: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmull_high_n_s32(int32x4_t a, int32_t b) {
  // CHECK: test_vqdmull_high_n_s32
  return vqdmull_high_n_s32(a, b);
  // CHECK: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlal_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  // CHECK: test_vmlal_high_n_s16
  return vmlal_high_n_s16(a, b, c);
  // CHECK: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlal_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  // CHECK: test_vmlal_high_n_s32
  return vmlal_high_n_s32(a, b, c);
  // CHECK: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmlal_high_n_u16(uint32x4_t a, uint16x8_t b, uint16_t c) {
  // CHECK: test_vmlal_high_n_u16
  return vmlal_high_n_u16(a, b, c);
  // CHECK: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmlal_high_n_u32(uint64x2_t a, uint32x4_t b, uint32_t c) {
  // CHECK: test_vmlal_high_n_u32
  return vmlal_high_n_u32(a, b, c);
  // CHECK: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlal_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  // CHECK: test_vqdmlal_high_n_s16
  return vqdmlal_high_n_s16(a, b, c);
  // CHECK: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlal_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  // CHECK: test_vqdmlal_high_n_s32
  return vqdmlal_high_n_s32(a, b, c);
  // CHECK: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vmlsl_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  // CHECK: test_vmlsl_high_n_s16
  return vmlsl_high_n_s16(a, b, c);
  // CHECK: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vmlsl_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  // CHECK: test_vmlsl_high_n_s32
  return vmlsl_high_n_s32(a, b, c);
  // CHECK: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

uint32x4_t test_vmlsl_high_n_u16(uint32x4_t a, uint16x8_t b, uint16_t c) {
  // CHECK: test_vmlsl_high_n_u16
  return vmlsl_high_n_u16(a, b, c);
  // CHECK: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

uint64x2_t test_vmlsl_high_n_u32(uint64x2_t a, uint32x4_t b, uint32_t c) {
  // CHECK: test_vmlsl_high_n_u32
  return vmlsl_high_n_u32(a, b, c);
  // CHECK: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

int32x4_t test_vqdmlsl_high_n_s16(int32x4_t a, int16x8_t b, int16_t c) {
  // CHECK: test_vqdmlsl_high_n_s16
  return vqdmlsl_high_n_s16(a, b, c);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, {{v[0-9]+}}.h[0]
}

int64x2_t test_vqdmlsl_high_n_s32(int64x2_t a, int32x4_t b, int32_t c) {
  // CHECK: test_vqdmlsl_high_n_s32
  return vqdmlsl_high_n_s32(a, b, c);
  // CHECK: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vmul_n_f32(float32x2_t a, float32_t b) {
  // CHECK: test_vmul_n_f32
  return vmul_n_f32(a, b);
  // CHECK: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vmulq_n_f32(float32x4_t a, float32_t b) {
  // CHECK: test_vmulq_n_f32
  return vmulq_n_f32(a, b);
  // CHECK: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float64x2_t test_vmulq_n_f64(float64x2_t a, float64_t b) {
  // CHECK: test_vmulq_n_f64
  return vmulq_n_f64(a, b);
  // CHECK: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

float32x2_t test_vfma_n_f32(float32x2_t a, float32x2_t b, float32_t n) {
  // CHECK: test_vfma_n_f32
  return vfma_n_f32(a, b, n);
  // CHECK: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmaq_n_f32(float32x4_t a, float32x4_t b, float32_t n) {
  // CHECK: test_vfmaq_n_f32
  return vfmaq_n_f32(a, b, n);
  // CHECK: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}

float32x2_t test_vfms_n_f32(float32x2_t a, float32x2_t b, float32_t n) {
  // CHECK: test_vfms_n_f32
  return vfms_n_f32(a, b, n);
  // CHECK: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
}

float32x4_t test_vfmsq_n_f32(float32x4_t a, float32x4_t b, float32_t n) {
  // CHECK: test_vfmsq_n_f32
  return vfmsq_n_f32(a, b, n);
  // CHECK: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
}
