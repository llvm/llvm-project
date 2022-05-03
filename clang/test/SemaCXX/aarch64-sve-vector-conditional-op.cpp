// RUN: %clang_cc1 -verify -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

void cond(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
          svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
          svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
          svbool_t b) {
  (void) i8 < i8 ? i16 : i16; // expected-error{{invalid operands to binary expression}}
  (void) i8 < i8 ? i32 : i32; // expected-error{{invalid operands to binary expression}}
  (void) i8 < i8 ? i64 : i64; // expected-error{{invalid operands to binary expression}}

  (void) i16 < i16 ? i16 : i8; // expected-error{{invalid operands to binary expression}}
  (void) i16 < i16 ? i16 : i32; // expected-error{{invalid operands to binary expression}}
  (void) i16 < i16 ? i16 : i64; // expected-error{{invalid operands to binary expression}}

  (void) i16 < i16 ? i8 : i16; // expected-error{{invalid operands to binary expression}}
  (void) i16 < i16 ? i32 : i16; // expected-error{{invalid operands to binary expression}}
  (void) i16 < i16 ? i64 : i16; // expected-error{{invalid operands to binary expression}}
}