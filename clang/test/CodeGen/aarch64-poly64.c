// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -ffp-contract=fast -S -O3 -o - %s | FileCheck %s --check-prefix=CHECK \
// RUN:  --check-prefix=CHECK-ARM64

// Test new aarch64 intrinsics with poly64

#include <arm_neon.h>

uint64x1_t test_vceq_p64(poly64x1_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vceq_p64
  return vceq_p64(a, b);
  // CHECK: cmeq {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x2_t test_vceqq_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vceqq_p64
  return vceqq_p64(a, b);
  // CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

uint64x1_t test_vtst_p64(poly64x1_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vtst_p64
  return vtst_p64(a, b);
  // CHECK: cmtst {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
}

uint64x2_t test_vtstq_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vtstq_p64
  return vtstq_p64(a, b);
  // CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x1_t test_vbsl_p64(poly64x1_t a, poly64x1_t b, poly64x1_t c) {
  // CHECK-LABEL: test_vbsl_p64
  return vbsl_p64(a, b, c);
  // CHECK: bsl {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
}

poly64x2_t test_vbslq_p64(poly64x2_t a, poly64x2_t b, poly64x2_t c) {
  // CHECK-LABEL: test_vbslq_p64
  return vbslq_p64(a, b, c);
  // CHECK: bsl {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
}

poly64_t test_vget_lane_p64(poly64x1_t v) {
  // CHECK-LABEL: test_vget_lane_p64
  return vget_lane_p64(v, 0);
  // CHECK: fmov  {{x[0-9]+}}, {{d[0-9]+}}
}

poly64_t test_vgetq_lane_p64(poly64x2_t v) {
  // CHECK-LABEL: test_vgetq_lane_p64
  return vgetq_lane_p64(v, 1);
  // CHECK: {{mov|umov}}  {{x[0-9]+}}, {{v[0-9]+}}.d[1]
}

poly64x1_t test_vset_lane_p64(poly64_t a, poly64x1_t v) {
  // CHECK-LABEL: test_vset_lane_p64
  return vset_lane_p64(a, v, 0);
  // CHECK: fmov  {{d[0-9]+}}, {{x[0-9]+}}
}

poly64x2_t test_vsetq_lane_p64(poly64_t a, poly64x2_t v) {
  // CHECK-LABEL: test_vsetq_lane_p64
  return vsetq_lane_p64(a, v, 1);
  // CHECK: ins  {{v[0-9]+}}.d[1], {{x[0-9]+}}
}

poly64x1_t test_vcopy_lane_p64(poly64x1_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vcopy_lane_p64
  return vcopy_lane_p64(a, 0, b, 0);

  // CHECK-ARM64: mov v0.16b, v1.16b
}

poly64x2_t test_vcopyq_lane_p64(poly64x2_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vcopyq_lane_p64
  return vcopyq_lane_p64(a, 1, b, 0);
  // CHECK: ins  {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
}

poly64x2_t test_vcopyq_laneq_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vcopyq_laneq_p64
  return vcopyq_laneq_p64(a, 1, b, 1);
}

poly64x1_t test_vcreate_p64(uint64_t a) {
  // CHECK-LABEL: test_vcreate_p64
  return vcreate_p64(a);
  // CHECK: fmov  {{d[0-9]+}}, {{x[0-9]+}}
}

poly64x1_t test_vdup_n_p64(poly64_t a) {
  // CHECK-LABEL: test_vdup_n_p64
  return vdup_n_p64(a);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}
poly64x2_t test_vdupq_n_p64(poly64_t a) {
  // CHECK-LABEL: test_vdupq_n_p64
  return vdupq_n_p64(a);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

poly64x1_t test_vmov_n_p64(poly64_t a) {
  // CHECK-LABEL: test_vmov_n_p64
  return vmov_n_p64(a);
  // CHECK: fmov {{d[0-9]+}}, {{x[0-9]+}}
}

poly64x2_t test_vmovq_n_p64(poly64_t a) {
  // CHECK-LABEL: test_vmovq_n_p64
  return vmovq_n_p64(a);
  // CHECK: dup {{v[0-9]+}}.2d, {{x[0-9]+}}
}

poly64x1_t test_vdup_lane_p64(poly64x1_t vec) {
  // CHECK-LABEL: test_vdup_lane_p64
  return vdup_lane_p64(vec, 0);
  // CHECK: ret
}

poly64x2_t test_vdupq_lane_p64(poly64x1_t vec) {
  // CHECK-LABEL: test_vdupq_lane_p64
  return vdupq_lane_p64(vec, 0);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
}

poly64x2_t test_vdupq_laneq_p64(poly64x2_t vec) {
  // CHECK-LABEL: test_vdupq_laneq_p64
  return vdupq_laneq_p64(vec, 1);
  // CHECK: dup {{v[0-9]+}}.2d, {{v[0-9]+}}.d[1]
}

poly64x2_t test_vcombine_p64(poly64x1_t low, poly64x1_t high) {
  // CHECK-LABEL: test_vcombine_p64
  return vcombine_p64(low, high);
  // CHECK: ins {{v[0-9]+}}.d[1], {{v[0-9]+}}.d[0]
}

poly64x1_t test_vld1_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld1_p64
  return vld1_p64(ptr);
  // CHECK-ARM64: ldr {{d[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly64x2_t test_vld1q_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld1q_p64
  return vld1q_p64(ptr);
  // CHECK-ARM64: ldr {{q[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1_p64(poly64_t * ptr, poly64x1_t val) {
  // CHECK-LABEL: test_vst1_p64
  return vst1_p64(ptr, val);
  // CHECK-ARM64: str {{d[0-9]+}}, [{{x[0-9]+|sp}}]
}

void test_vst1q_p64(poly64_t * ptr, poly64x2_t val) {
  // CHECK-LABEL: test_vst1q_p64
  return vst1q_p64(ptr, val);
  // CHECK-ARM64: str {{q[0-9]+}}, [{{x[0-9]+|sp}}]
}

poly64x1x2_t test_vld2_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld2_p64
  return vld2_p64(ptr);
  // CHECK: ld1 {{{ *v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

poly64x2x2_t test_vld2q_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld2q_p64
  return vld2q_p64(ptr);
  // CHECK: ld2 {{{ *v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

poly64x1x3_t test_vld3_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld3_p64
  return vld3_p64(ptr);
  // CHECK: ld1 {{{ *v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

poly64x2x3_t test_vld3q_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld3q_p64
  return vld3q_p64(ptr);
  // CHECK: ld3 {{{ *v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

poly64x1x4_t test_vld4_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld4_p64
  return vld4_p64(ptr);
  // CHECK: ld1 {{{ *v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

poly64x2x4_t test_vld4q_p64(poly64_t const * ptr) {
  // CHECK-LABEL: test_vld4q_p64
  return vld4q_p64(ptr);
  // CHECK: ld4 {{{ *v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst2_p64(poly64_t * ptr, poly64x1x2_t val) {
  // CHECK-LABEL: test_vst2_p64
  return vst2_p64(ptr, val);
  // CHECK:  st1 {{{ *v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst2q_p64(poly64_t * ptr, poly64x2x2_t val) {
  // CHECK-LABEL: test_vst2q_p64
  return vst2q_p64(ptr, val);
  // CHECK:  st2 {{{ *v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst3_p64(poly64_t * ptr, poly64x1x3_t val) {
  // CHECK-LABEL: test_vst3_p64
  return vst3_p64(ptr, val);
  // CHECK:  st1 {{{ *v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst3q_p64(poly64_t * ptr, poly64x2x3_t val) {
  // CHECK-LABEL: test_vst3q_p64
  return vst3q_p64(ptr, val);
  // CHECK:  st3 {{{ *v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst4_p64(poly64_t * ptr, poly64x1x4_t val) {
  // CHECK-LABEL: test_vst4_p64
  return vst4_p64(ptr, val);
  // CHECK:  st1 {{{ *v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d, v[0-9]+.1d *}}}, [{{x[0-9]+|sp}}]
}

void test_vst4q_p64(poly64_t * ptr, poly64x2x4_t val) {
  // CHECK-LABEL: test_vst4q_p64
  return vst4q_p64(ptr, val);
  // CHECK:  st4 {{{ *v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d, v[0-9]+.2d *}}}, [{{x[0-9]+|sp}}]
}

poly64x1_t test_vext_p64(poly64x1_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vext_p64
  return vext_u64(a, b, 0);

}

poly64x2_t test_vextq_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vextq_p64
  return vextq_p64(a, b, 1);
  // CHECK: ext {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{#0x8|#8}}
}

poly64x2_t test_vzip1q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vzip1q_p64
  return vzip1q_p64(a, b);
  // CHECK-ARM64: zip1 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x2_t test_vzip2q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vzip2q_p64
  return vzip2q_u64(a, b);
  // CHECK-ARM64: zip2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x2_t test_vuzp1q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vuzp1q_p64
  return vuzp1q_p64(a, b);
  // CHECK-ARM64: zip1 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x2_t test_vuzp2q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vuzp2q_p64
  return vuzp2q_u64(a, b);
  // CHECK-ARM64: zip2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x2_t test_vtrn1q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vtrn1q_p64
  return vtrn1q_p64(a, b);
  // CHECK-ARM64: zip1 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x2_t test_vtrn2q_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vtrn2q_p64
  return vtrn2q_u64(a, b);
  // CHECK-ARM64: zip2 {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
}

poly64x1_t test_vsri_n_p64(poly64x1_t a, poly64x1_t b) {
  // CHECK-LABEL: test_vsri_n_p64
  return vsri_n_p64(a, b, 33);
  // CHECK: sri {{d[0-9]+}}, {{d[0-9]+}}, #33
}

poly64x2_t test_vsriq_n_p64(poly64x2_t a, poly64x2_t b) {
  // CHECK-LABEL: test_vsriq_n_p64
  return vsriq_n_p64(a, b, 64);
  // CHECK: sri {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #64
}

