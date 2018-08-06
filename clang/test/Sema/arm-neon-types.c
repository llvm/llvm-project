// RUN: %clang_cc1 -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -Wvector-conversion -ffreestanding -verify %s
#ifndef INCLUDE

#include <arm_neon.h>

// Radar 8228022: Should not report incompatible vector types.
int32x2_t test(int32x2_t x) {
  return vshr_n_s32(x, 31);
}

// ...but should warn when the types really do not match.
float32x2_t test2(uint32x2_t x) {
  return vcvt_n_f32_s32(x, 9); // expected-warning {{incompatible vector types}}
}

// Check immediate range for vcvt_n intrinsics is 1 to 32.  Radar 9558930.
float32x2_t test3(uint32x2_t x) {
  // FIXME: The "incompatible result type" error is due to pr10112 and should be
  // removed when that is fixed.
  return vcvt_n_f32_u32(x, 0); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

typedef signed int vSInt32 __attribute__((__vector_size__(16)));
int32x4_t test4(int32x4_t a, vSInt32 b) {
  a += b;
  b += a;
  return b += a;
}

// Warn for incompatible pointer types used with vld/vst intrinsics.
int16x8_t test5(int *p) {
  return vld1q_s16(p); // expected-warning {{incompatible pointer types}}
}
void test6(float *p, int32x2_t v) {
  return vst1_s32(p, v); // expected-warning {{incompatible pointer types}}
}

#define INCLUDE
#include "arm-neon-types.c"
#else

// Make sure we don't get a warning about using a static function in an
// extern inline function from a header.
extern inline uint8x8_t test7(uint8x8_t a, uint8x8_t b) {
  return vadd_u8(a, b);
}
#endif
