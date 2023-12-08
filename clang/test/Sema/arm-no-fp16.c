// RUN: %clang_cc1 -triple thumbv7-none-eabi %s -target-feature +neon \
// RUN:   -target-feature -fp16 \
// RUN:   -fsyntax-only -verify -Wno-error=implicit-function-declaration

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  return vcvt_f16_f32(a); // expected-warning{{call to undeclared function 'vcvt_f16_f32'}}  expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  return vcvt_f32_f16(a); // expected-warning{{call to undeclared function 'vcvt_f32_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float32x4_t'}}
}

float16x4_t test_vrnda_f16(float16x4_t a) {
  return vrnda_f16(a); // expected-warning{{call to undeclared function 'vrnda_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndaq_f16(float16x8_t a) {
  return vrndaq_f16(a); // expected-warning{{call to undeclared function 'vrndaq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrnd_f16(float16x4_t a) {
  return vrnd_f16(a); // expected-warning{{call to undeclared function 'vrnd_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndq_f16(float16x8_t a) {
  return vrndq_f16(a); // expected-warning{{call to undeclared function 'vrndq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndi_f16(float16x4_t a) {
  return vrndi_f16(a); // expected-warning{{call to undeclared function 'vrndi_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndiq_f16(float16x8_t a) {
  return vrndiq_f16(a); // expected-warning{{call to undeclared function 'vrndiq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndm_f16(float16x4_t a) {
  return vrndm_f16(a); // expected-warning{{call to undeclared function 'vrndm_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndmq_f16(float16x8_t a) {
  return vrndmq_f16(a); // expected-warning{{call to undeclared function 'vrndmq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndn_f16(float16x4_t a) {
  return vrndn_f16(a); // expected-warning{{call to undeclared function 'vrndn_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndnq_f16(float16x8_t a) {
  return vrndnq_f16(a); // expected-warning{{call to undeclared function 'vrndnq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndp_f16(float16x4_t a) {
  return vrndp_f16(a); // expected-warning{{call to undeclared function 'vrndp_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndpq_f16(float16x8_t a) {
  return vrndpq_f16(a); // expected-warning{{call to undeclared function 'vrndpq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vrndx_f16(float16x4_t a) {
  return vrndx_f16(a); // expected-warning{{call to undeclared function 'vrndx_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vrndxq_f16(float16x8_t a) {
  return vrndxq_f16(a); // expected-warning{{call to undeclared function 'vrndxq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vmaxnm_f16(float16x4_t a, float16x4_t b) {
  return vmaxnm_f16(a, b); // expected-warning{{call to undeclared function 'vmaxnm_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vmaxnmq_f16(float16x8_t a, float16x8_t b) {
  return vmaxnmq_f16(a, b); // expected-warning{{call to undeclared function 'vmaxnmq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vminnm_f16(float16x4_t a, float16x4_t b) {
  return vminnm_f16(a, b); // expected-warning{{call to undeclared function 'vminnm_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vminnmq_f16(float16x8_t a, float16x8_t b) {
  return vminnmq_f16(a, b); // expected-warning{{call to undeclared function 'vminnmq_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vld1_f16(const float16_t *a) {
  return vld1_f16(a); // expected-warning{{call to undeclared function 'vld1_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vld1q_f16(const float16_t *a) {
  return vld1q_f16(a); // expected-warning{{call to undeclared function 'vld1q_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vld1_dup_f16(const float16_t *a) {
  return vld1_dup_f16(a); // expected-warning{{call to undeclared function 'vld1_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vld1q_dup_f16(const float16_t *a) {
  return vld1q_dup_f16(a); // expected-warning{{call to undeclared function 'vld1q_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4_t test_vld1_lane_f16(const float16_t *a, float16x4_t b) {
  return vld1_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vld1_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float16x8_t test_vld1q_lane_f16(const float16_t *a, float16x8_t b) {
  return vld1q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vld1q_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8_t'}}
}

float16x4x2_t test_vld1_f16_x2(const float16_t *a) {
  return vld1_f16_x2(a); // expected-warning{{call to undeclared function 'vld1_f16_x2'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x2_t'}}
}

float16x8x2_t test_vld1q_f16_x2(const float16_t *a) {
  return vld1q_f16_x2(a); // expected-warning{{call to undeclared function 'vld1q_f16_x2'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x2_t'}}
}

float16x4x3_t test_vld1_f16_x3(const float16_t *a) {
  return vld1_f16_x3(a); // expected-warning{{call to undeclared function 'vld1_f16_x3'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x3_t'}}
}

float16x8x3_t test_vld1q_f16_x3(const float16_t *a) {
  return vld1q_f16_x3(a); // expected-warning{{call to undeclared function 'vld1q_f16_x3'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x3_t'}}
}

float16x4x4_t test_vld1_f16_x4(const float16_t *a) {
  return vld1_f16_x4(a); // expected-warning{{call to undeclared function 'vld1_f16_x4'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x4_t'}}
}

float16x8x4_t test_vld1q_f16_x4(const float16_t *a) {
  return vld1q_f16_x4(a); // expected-warning{{call to undeclared function 'vld1q_f16_x4'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x4_t'}}
}

float16x4x2_t test_vld2_f16(const float16_t *a) {
  return vld2_f16(a); // expected-warning{{call to undeclared function 'vld2_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x2_t'}}
}

float16x8x2_t test_vld2q_f16(const float16_t *a) {
  return vld2q_f16(a); // expected-warning{{call to undeclared function 'vld2q_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x2_t'}}
}

float16x4x2_t test_vld2_lane_f16(const float16_t *a, float16x4x2_t b) {
  return vld2_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vld2_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x2_t'}}
}

float16x8x2_t test_vld2q_lane_f16(const float16_t *a, float16x8x2_t b) {
  return vld2q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vld2q_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x2_t'}}
}

float16x4x2_t test_vld2_dup_f16(const float16_t *src) {
  return vld2_dup_f16(src); // expected-warning{{call to undeclared function 'vld2_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x2_t'}}
}

float16x8x2_t test_vld2q_dup_f16(const float16_t *src) {
  return vld2q_dup_f16(src); // expected-warning{{call to undeclared function 'vld2q_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x2_t'}}
}

float16x4x3_t test_vld3_f16(const float16_t *a) {
  return vld3_f16(a); // expected-warning{{call to undeclared function 'vld3_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x3_t'}}
}

float16x8x3_t test_vld3q_f16(const float16_t *a) {
  return vld3q_f16(a); // expected-warning{{call to undeclared function 'vld3q_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x3_t'}}
}

float16x4x3_t test_vld3_lane_f16(const float16_t *a, float16x4x3_t b) {
  return vld3_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vld3_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x3_t'}}
}

float16x8x3_t test_vld3q_lane_f16(const float16_t *a, float16x8x3_t b) {
  return vld3q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vld3q_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x3_t'}}
}

float16x4x3_t test_vld3_dup_f16(const float16_t *src) {
  return vld3_dup_f16(src); // expected-warning{{call to undeclared function 'vld3_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x3_t'}}
}

float16x8x3_t test_vld3q_dup_f16(const float16_t *src) {
  return vld3q_dup_f16(src); // expected-warning{{call to undeclared function 'vld3q_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x3_t'}}
}

float16x4x4_t test_vld4_f16(const float16_t *a) {
  return vld4_f16(a); // expected-warning{{call to undeclared function 'vld4_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x4_t'}}
}

float16x8x4_t test_vld4q_f16(const float16_t *a) {
  return vld4q_f16(a); // expected-warning{{call to undeclared function 'vld4q_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x4_t'}}
}

float16x4x4_t test_vld4_lane_f16(const float16_t *a, float16x4x4_t b) {
  return vld4_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vld4_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x4_t'}}
}

float16x8x4_t test_vld4q_lane_f16(const float16_t *a, float16x8x4_t b) {
  return vld4q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vld4q_lane_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x4_t'}}
}

float16x4x4_t test_vld4_dup_f16(const float16_t *src) {
  return vld4_dup_f16(src); // expected-warning{{call to undeclared function 'vld4_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x4x4_t'}}
}

float16x8x4_t test_vld4q_dup_f16(const float16_t *src) {
  return vld4q_dup_f16(src); // expected-warning{{call to undeclared function 'vld4q_dup_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float16x8x4_t'}}
}

void test_vst1_f16(float16_t *a, float16x4_t b) {
  vst1_f16(a, b); // expected-warning{{call to undeclared function 'vst1_f16'}}
}

// aarch64-neon-intrinsics.c:void test_vst1q_f16(float16_t *a, float16x8_t b) {
void test_vst1q_f16(float16_t *a, float16x8_t b) {
  vst1q_f16(a, b); // expected-warning{{call to undeclared function 'vst1q_f16'}}
}

// aarch64-neon-ldst-one.c:void test_vst1_lane_f16(float16_t  *a, float16x4_t b) {
void test_vst1_lane_f16(float16_t *a, float16x4_t b) {
  vst1_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vst1_lane_f16'}}
}

void test_vst1q_lane_f16(float16_t *a, float16x8_t b) {
  vst1q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vst1q_lane_f16'}}
}

void test_vst1_f16_x2(float16_t *a, float16x4x2_t b) {
  vst1_f16_x2(a, b); // expected-warning{{call to undeclared function 'vst1_f16_x2'}}
}

void test_vst1q_f16_x2(float16_t *a, float16x8x2_t b) {
  vst1q_f16_x2(a, b); // expected-warning{{call to undeclared function 'vst1q_f16_x2'}}
}

void test_vst1_f16_x3(float16_t *a, float16x4x3_t b) {
  vst1_f16_x3(a, b); // expected-warning{{call to undeclared function 'vst1_f16_x3'}}
}

void test_vst1q_f16_x3(float16_t *a, float16x8x3_t b) {
  vst1q_f16_x3(a, b); // expected-warning{{call to undeclared function 'vst1q_f16_x3'}}
}

void test_vst1_f16_x4(float16_t *a, float16x4x4_t b) {
  vst1_f16_x4(a, b); // expected-warning{{call to undeclared function 'vst1_f16_x4'}}
}

void test_vst1q_f16_x4(float16_t *a, float16x8x4_t b) {
  vst1q_f16_x4(a, b); // expected-warning{{call to undeclared function 'vst1q_f16_x4'}}
}

void test_vst2_f16(float16_t *a, float16x4x2_t b) {
  vst2_f16(a, b); // expected-warning{{call to undeclared function 'vst2_f16'}}
}

void test_vst2q_f16(float16_t *a, float16x8x2_t b) {
  vst2q_f16(a, b); // expected-warning{{call to undeclared function 'vst2q_f16'}}
}

void test_vst2_lane_f16(float16_t *a, float16x4x2_t b) {
  vst2_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vst2_lane_f16'}}
}

void test_vst2q_lane_f16(float16_t *a, float16x8x2_t b) {
  vst2q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vst2q_lane_f16'}}
}

void test_vst3_f16(float16_t *a, float16x4x3_t b) {
  vst3_f16(a, b); // expected-warning{{call to undeclared function 'vst3_f16'}}
}

void test_vst3q_f16(float16_t *a, float16x8x3_t b) {
  vst3q_f16(a, b); // expected-warning{{call to undeclared function 'vst3q_f16'}}
}

void test_vst3_lane_f16(float16_t *a, float16x4x3_t b) {
  vst3_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vst3_lane_f16'}}
}

void test_vst3q_lane_f16(float16_t *a, float16x8x3_t b) {
  vst3q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vst3q_lane_f16'}}
}

void test_vst4_f16(float16_t *a, float16x4x4_t b) {
  vst4_f16(a, b); // expected-warning{{call to undeclared function 'vst4_f16'}}
}

void test_vst4q_f16(float16_t *a, float16x8x4_t b) {
  vst4q_f16(a, b); // expected-warning{{call to undeclared function 'vst4q_f16'}}
}

void test_vst4_lane_f16(float16_t *a, float16x4x4_t b) {
  vst4_lane_f16(a, b, 3); // expected-warning{{call to undeclared function 'vst4_lane_f16'}}
}

void test_vst4q_lane_f16(float16_t *a, float16x8x4_t b) {
  vst4q_lane_f16(a, b, 7); // expected-warning{{call to undeclared function 'vst4q_lane_f16'}}
}
