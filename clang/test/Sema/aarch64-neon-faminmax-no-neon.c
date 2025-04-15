// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +faminmax -emit-llvm -verify %s -o /dev/null

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

float16x4_t a16x4, b16x4;
float16x8_t a16x8, b16x8;
float32x2_t a32x2, b32x2;
float32x4_t a32x4, b32x4;
float64x2_t a64x2, b64x2;

void test () {
  (void) vamin_f16 (a16x4, b16x4);
// expected-error@-1 {{always_inline function 'vamin_f16' requires target feature 'neon'}}
  (void) vaminq_f16(a16x8, b16x8);
// expected-error@-1 {{always_inline function 'vaminq_f16' requires target feature 'neon'}}
  (void) vamin_f32 (a32x2, b32x2);
// expected-error@-1 {{always_inline function 'vamin_f32' requires target feature 'neon'}}
  (void) vaminq_f32(a32x4, b32x4);
// expected-error@-1 {{always_inline function 'vaminq_f32' requires target feature 'neon'}}
  (void) vaminq_f64(a64x2, b64x2);
// expected-error@-1 {{always_inline function 'vaminq_f64' requires target feature 'neon'}}
  (void) vamax_f16 (a16x4, b16x4);
// expected-error@-1 {{always_inline function 'vamax_f16' requires target feature 'neon'}}
  (void) vamaxq_f16(a16x8, b16x8);
// expected-error@-1 {{always_inline function 'vamaxq_f16' requires target feature 'neon'}}
  (void) vamax_f32 (a32x2, b32x2);
// expected-error@-1 {{always_inline function 'vamax_f32' requires target feature 'neon'}}
  (void) vamaxq_f32(a32x4, b32x4);
// expected-error@-1 {{always_inline function 'vamaxq_f32' requires target feature 'neon'}}
  (void) vamaxq_f64(a64x2, b64x2);
// expected-error@-1 {{always_inline function 'vamaxq_f64' requires target feature 'neon'}}
}
