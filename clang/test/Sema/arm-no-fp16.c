// RUN: %clang_cc1 -triple thumbv7-none-eabi %s -target-feature +neon -target-feature -fp16 -fsyntax-only -verify

#include <arm_neon.h>

float16x4_t test_vcvt_f16_f32(float32x4_t a) {
  return vcvt_f16_f32(a); // expected-warning{{implicit declaration of function 'vcvt_f16_f32'}}  expected-error{{returning 'int' from a function with incompatible result type 'float16x4_t'}}
}

float32x4_t test_vcvt_f32_f16(float16x4_t a) {
  return vcvt_f32_f16(a); // expected-warning{{implicit declaration of function 'vcvt_f32_f16'}} expected-error{{returning 'int' from a function with incompatible result type 'float32x4_t'}}
}
