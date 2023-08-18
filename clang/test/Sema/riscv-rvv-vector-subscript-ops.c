// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

void subscript(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
               vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
               vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
               vbool1_t b) {
  (void)b[0.f];  // expected-error{{array subscript is not an integer}}
  (void)b[0.];   // expected-error{{array subscript is not an integer}}

  (void)i8[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i8[0.];  // expected-error{{array subscript is not an integer}}

  (void)u8[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u8[0.];  // expected-error{{array subscript is not an integer}}

  (void)i16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i16[0.];  // expected-error{{array subscript is not an integer}}

  (void)u16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u16[0.];  // expected-error{{array subscript is not an integer}}

  (void)i32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i32[0.];  // expected-error{{array subscript is not an integer}}

  (void)u32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u32[0.];  // expected-error{{array subscript is not an integer}}

  (void)i64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)i64[0.];  // expected-error{{array subscript is not an integer}}

  (void)u64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)u64[0.];  // expected-error{{array subscript is not an integer}}

  (void)f16[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f16[0.];  // expected-error{{array subscript is not an integer}}

  (void)f32[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f32[0.];  // expected-error{{array subscript is not an integer}}

  (void)f64[0.f]; // expected-error{{array subscript is not an integer}}
  (void)f64[0.];  // expected-error{{array subscript is not an integer}}
}
