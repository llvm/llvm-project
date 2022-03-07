// RUN: %clang_cc1 -verify -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

void and (svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
          svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
          svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
          svbool_t b) {
  (void)(i8 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(u8 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(i16 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(u16 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(i32 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(i32 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(u32 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(u32 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(i64 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(u64 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(f16 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(f32 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & u64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 & f64); // expected-error{{invalid operands to binary expression}}

  (void)(f64 & b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 & i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 & i16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & i32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & i64); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 & u16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & u32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & f16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & f32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 & f64); // expected-error{{invalid operands to binary expression}}
}

void or (svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b) {
  (void)(i8 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(u8 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(i16 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(u16 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(i32 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(i32 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(u32 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(u32 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(i64 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(u64 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(f16 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(f32 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | u64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 | f64); // expected-error{{invalid operands to binary expression}}

  (void)(f64 | b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 | i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 | i16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | i32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | i64); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 | u16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | u32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | f16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | f32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 | f64); // expected-error{{invalid operands to binary expression}}
}

void xor (svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64, svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64, svfloat16_t f16, svfloat32_t f32, svfloat64_t f64, svbool_t b) {
  (void)(i8 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(u8 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(i16 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(u16 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(i32 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(i32 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(u32 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(u32 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(i64 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(u64 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(f16 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(f32 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ u64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 ^ f64); // expected-error{{invalid operands to binary expression}}

  (void)(f64 ^ b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ i16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ i32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ i64); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ u16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ u32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ f16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ f32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 ^ f64); // expected-error{{invalid operands to binary expression}}
}

    void not(svfloat16_t f16, svfloat32_t f32, svfloat32_t f64) {
  (void)(~f16); // expected-error{{invalid argument type}}
  (void)(~f32); // expected-error{{invalid argument type}}
  (void)(~f64); // expected-error{{invalid argument type}}
}
