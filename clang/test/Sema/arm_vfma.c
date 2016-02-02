// RUN: %clang_cc1 -triple thumbv7s-apple-ios7.0 -target-feature +neon -fsyntax-only -verify %s
#include <arm_neon.h>

// expected-no-diagnostics

void func(float32x2_t v2f32, float32x4_t v4f32) {
  vfma_f32(v2f32, v2f32, v2f32);
  vfmaq_f32(v4f32, v4f32, v4f32);

  vfms_f32(v2f32, v2f32, v2f32);
  vfmsq_f32(v4f32, v4f32, v4f32);
}
