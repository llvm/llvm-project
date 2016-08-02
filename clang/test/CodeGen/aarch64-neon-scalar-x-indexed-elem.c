// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN:   -ffp-contract=fast -S -O3 -o - %s | FileCheck %s

// Test new aarch64 intrinsics and types

#include <arm_neon.h>


float32_t test_vmuls_lane_f32(float32_t a, float32x2_t b) {
  // CHECK-LABEL: test_vmuls_lane_f32
  return vmuls_lane_f32(a, b, 1);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float64_t test_vmuld_lane_f64(float64_t a, float64x1_t b) {
  // CHECK-LABEL: test_vmuld_lane_f64
  return vmuld_lane_f64(a, b, 0);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

float32_t test_vmuls_laneq_f32(float32_t a, float32x4_t b) {
  // CHECK-LABEL: test_vmuls_laneq_f32
  return vmuls_laneq_f32(a, b, 3);
  // CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmuld_laneq_f64(float64_t a, float64x2_t b) {
  // CHECK-LABEL: test_vmuld_laneq_f64
  return vmuld_laneq_f64(a, b, 1);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

float64x1_t test_vmul_n_f64(float64x1_t a, float64_t b) {
  // CHECK-LABEL: test_vmul_n_f64
  return vmul_n_f64(a, b);
  // CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

float32_t test_vmulxs_lane_f32(float32_t a, float32x2_t b) {
// CHECK-LABEL: test_vmulxs_lane_f32
  return vmulxs_lane_f32(a, b, 1);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

float32_t test_vmulxs_laneq_f32(float32_t a, float32x4_t b) {
// CHECK-LABEL: test_vmulxs_laneq_f32
  return vmulxs_laneq_f32(a, b, 3);
// CHECK: fmulx {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

float64_t test_vmulxd_lane_f64(float64_t a, float64x1_t b) {
// CHECK-LABEL: test_vmulxd_lane_f64
  return vmulxd_lane_f64(a, b, 0);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

float64_t test_vmulxd_laneq_f64(float64_t a, float64x2_t b) {
// CHECK-LABEL: test_vmulxd_laneq_f64
  return vmulxd_laneq_f64(a, b, 1);
// CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK-LABEL: test_vmulx_lane_f64
float64x1_t test_vmulx_lane_f64(float64x1_t a, float64x1_t b) {
  return vmulx_lane_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}


// CHECK-LABEL: test_vmulx_laneq_f64_0
float64x1_t test_vmulx_laneq_f64_0(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 0);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK-LABEL: test_vmulx_laneq_f64_1
float64x1_t test_vmulx_laneq_f64_1(float64x1_t a, float64x2_t b) {
  return vmulx_laneq_f64(a, b, 1);
  // CHECK: fmulx {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}


// CHECK-LABEL: test_vfmas_lane_f32
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
  // CHECK: fmla {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK-LABEL: test_vfmad_lane_f64
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
  // CHECK: {{fmla|fmadd}} {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

// CHECK-LABEL: test_vfmad_laneq_f64
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[1]
}

// CHECK-LABEL: test_vfmss_lane_f32
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmss_lane_f32(a, b, c, 1);
  // CHECK: fmls {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK-LABEL: test_vfma_lane_f64
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
  // CHECK: {{fmla|fmadd}} {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

// CHECK-LABEL: test_vfms_lane_f64
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfms_lane_f64(a, b, v, 0);
  // CHECK: {{fmls|fmsub}} {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+.d\[0\]|d[0-9]+}}
}

// CHECK-LABEL: test_vfma_laneq_f64
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
  // CHECK: fmla {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK-LABEL: test_vfms_laneq_f64
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfms_laneq_f64(a, b, v, 0);
  // CHECK: fmls {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}.d[0]
}

// CHECK-LABEL: test_vqdmullh_lane_s16
int32_t test_vqdmullh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmullh_lane_s16(a, b, 3);
  // CHECK: sqdmull {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9].4h}}, {{v[0-9]+}}.h[3]
}

// CHECK-LABEL: test_vqdmulls_lane_s32
int64_t test_vqdmulls_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulls_lane_s32(a, b, 1);
  // CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK-LABEL: test_vqdmullh_laneq_s16
int32_t test_vqdmullh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmullh_laneq_s16(a, b, 7);
  // CHECK: sqdmull {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
}

// CHECK-LABEL: test_vqdmulls_laneq_s32
int64_t test_vqdmulls_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulls_laneq_s32(a, b, 3);
  // CHECK: sqdmull {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK-LABEL: test_vqdmulhh_lane_s16
int16_t test_vqdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqdmulhh_lane_s16(a, b, 3);
// CHECK: sqdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
}

// CHECK-LABEL: test_vqdmulhs_lane_s32
int32_t test_vqdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqdmulhs_lane_s32(a, b, 1);
// CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK-LABEL: test_vqdmulhh_laneq_s16
int16_t test_vqdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqdmulhh_laneq_s16(a, b, 7);
// CHECK: sqdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
}


// CHECK-LABEL: test_vqdmulhs_laneq_s32
int32_t test_vqdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqdmulhs_laneq_s32(a, b, 3);
// CHECK: sqdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK-LABEL: test_vqrdmulhh_lane_s16
int16_t test_vqrdmulhh_lane_s16(int16_t a, int16x4_t b) {
  return vqrdmulhh_lane_s16(a, b, 3);
// CHECK: sqrdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
}

// CHECK-LABEL: test_vqrdmulhs_lane_s32
int32_t test_vqrdmulhs_lane_s32(int32_t a, int32x2_t b) {
  return vqrdmulhs_lane_s32(a, b, 1);
// CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}


// CHECK-LABEL: test_vqrdmulhh_laneq_s16
int16_t test_vqrdmulhh_laneq_s16(int16_t a, int16x8_t b) {
  return vqrdmulhh_laneq_s16(a, b, 7);
// CHECK: sqrdmulh {{h[0-9]+|v[0-9]+.4h}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
}


// CHECK-LABEL: test_vqrdmulhs_laneq_s32
int32_t test_vqrdmulhs_laneq_s32(int32_t a, int32x4_t b) {
  return vqrdmulhs_laneq_s32(a, b, 3);
// CHECK: sqrdmulh {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK-LABEL: test_vqdmlalh_lane_s16
int32_t test_vqdmlalh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlalh_lane_s16(a, b, c, 3);
// CHECK: sqdmlal {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
}

// CHECK-LABEL: test_vqdmlals_lane_s32
int64_t test_vqdmlals_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlals_lane_s32(a, b, c, 1);
// CHECK: sqdmlal {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK-LABEL: test_vqdmlalh_laneq_s16
int32_t test_vqdmlalh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlalh_laneq_s16(a, b, c, 7);
// CHECK: sqdmlal {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
}

// CHECK-LABEL: test_vqdmlals_laneq_s32
int64_t test_vqdmlals_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlals_laneq_s32(a, b, c, 3);
// CHECK: sqdmlal {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK-LABEL: test_vqdmlslh_lane_s16
int32_t test_vqdmlslh_lane_s16(int32_t a, int16_t b, int16x4_t c) {
  return vqdmlslh_lane_s16(a, b, c, 3);
// CHECK: sqdmlsl {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[3]
}

// CHECK-LABEL: test_vqdmlsls_lane_s32
int64_t test_vqdmlsls_lane_s32(int64_t a, int32_t b, int32x2_t c) {
  return vqdmlsls_lane_s32(a, b, c, 1);
// CHECK: sqdmlsl {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[1]
}

// CHECK-LABEL: test_vqdmlslh_laneq_s16
int32_t test_vqdmlslh_laneq_s16(int32_t a, int16_t b, int16x8_t c) {
  return vqdmlslh_laneq_s16(a, b, c, 7);
// CHECK: sqdmlsl {{s[0-9]+|v[0-9]+.4s}}, {{h[0-9]+|v[0-9]+.4h}}, {{v[0-9]+}}.h[7]
}

// CHECK-LABEL: test_vqdmlsls_laneq_s32
int64_t test_vqdmlsls_laneq_s32(int64_t a, int32_t b, int32x4_t c) {
  return vqdmlsls_laneq_s32(a, b, c, 3);
// CHECK: sqdmlsl {{d[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}.s[3]
}

// CHECK-LABEL: test_vmulx_lane_f64_0:
float64x1_t test_vmulx_lane_f64_0() {
      float64x1_t arg1;
      float64x1_t arg2;
      float64x1_t result;
      float64_t sarg1, sarg2, sres;
      arg1 = vcreate_f64(UINT64_C(0x3fd6304bc43ab5c2));
      arg2 = vcreate_f64(UINT64_C(0x3fee211e215aeef3));
      result = vmulx_lane_f64(arg1, arg2, 0);
// CHECK: adrp x[[ADDRLO:[0-9]+]]
// CHECK: ldr d0, [x[[ADDRLO]],
// CHECK: adrp x[[ADDRLO:[0-9]+]]
// CHECK: ldr d1, [x[[ADDRLO]],
// CHECK: fmulx d0, d1, d0
      return result;
}

// CHECK-LABEL: test_vmulx_laneq_f64_2:
float64x1_t test_vmulx_laneq_f64_2() {
      float64x1_t arg1;
      float64x1_t arg2;
      float64x2_t arg3;
      float64x1_t result;
      float64_t sarg1, sarg2, sres;
      arg1 = vcreate_f64(UINT64_C(0x3fd6304bc43ab5c2));
      arg2 = vcreate_f64(UINT64_C(0x3fee211e215aeef3));
      arg3 = vcombine_f64(arg1, arg2);
      result = vmulx_laneq_f64(arg1, arg3, 1);
// CHECK: adrp x[[ADDRLO:[0-9]+]]
// CHECK: ldr d0, [x[[ADDRLO]],
// CHECK: adrp x[[ADDRLO:[0-9]+]]
// CHECK: ldr d1, [x[[ADDRLO]],
// CHECK: fmulx d0, d1, d0
      return result;
}
