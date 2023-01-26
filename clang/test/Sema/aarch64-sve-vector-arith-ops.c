// RUN: %clang_cc1 -verify -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

struct S {
  int x;
};

void add(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b, struct S s) {
  (void)(b + b); // expected-error{{invalid operands to binary expression}}

  (void)(i8 + b); // expected-error{{invalid operands to binary expression}}
  (void)(i8 + i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 + i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 + i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 + u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 + u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 + u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 + f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 + f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 + f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 + i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 + i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 + i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 + u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 + u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 + u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 + f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 + f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 + f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 + i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 + i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 + i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 + u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 + u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 + u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 + f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 + f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 + f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 + i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 + i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 + i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 + u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 + u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 + u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 + f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 + f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 + f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 + i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 + i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 + i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 + u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 + u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 + u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 + f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 + f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 + f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 + i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 + i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 + i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 + u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 + u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 + u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 + f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 + f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 + f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 + i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 + i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 + i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 + u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 + u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 + u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 + f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 + f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 + f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 + i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 + i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 + i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 + u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 + u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 + u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 + f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 + f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 + f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 + i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 + i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 + i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 + i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 + u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 + u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 + u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 + f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 + f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 + i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 + i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 + i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 + i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 + u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 + u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 + u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 + f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 + f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 + b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 + i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 + i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 + i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 + i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 + u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 + u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 + u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 + f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 + f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}

  (void)(s + i8); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + i16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + i32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + i64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + u16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + u32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + u64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + f16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + f32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s + f64); // expected-error{{cannot convert between vector and non-scalar values}}
}

void sub(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b, struct S s) {
  (void)(b - b); // expected-error{{invalid operands to binary expression}}

  (void)(i8 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 - i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 - i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 - i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 - u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 - u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 - u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 - f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 - f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 - f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 - i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 - i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 - i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 - u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 - u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 - u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 - f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 - f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 - f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 - i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 - i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 - i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 - u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 - u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 - u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 - f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 - f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 - f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 - i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 - i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 - i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 - u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 - u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 - u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 - f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 - f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 - f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 - i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 - i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 - i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 - u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 - u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 - u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 - f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 - f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 - f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 - i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 - i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 - i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 - u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 - u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 - u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 - f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 - f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 - f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 - i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 - i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 - i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 - u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 - u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 - u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 - f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 - f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 - f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 - i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 - i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 - i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 - u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 - u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 - u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 - f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 - f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 - f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 - i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 - i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 - i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 - i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 - u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 - u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 - u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 - f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 - f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 - i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 - i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 - i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 - i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 - u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 - u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 - u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 - f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 - f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 - b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 - i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 - i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 - i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 - i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 - u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 - u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 - u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 - f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 - f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}

  (void)(s - i8);   // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - i16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - i32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - i64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - u16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - u32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - u64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - f16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - f32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s - f64); // expected-error{{cannot convert between vector and non-scalar values}}
}

void mul(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b, struct S s) {
  (void)(b * b); // expected-error{{invalid operands to binary expression}}

  (void)(i8 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 * i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 * i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 * i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 * u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 * u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 * u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 * f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 * f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 * f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 * i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 * i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 * i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 * u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 * u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 * u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 * f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 * f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 * f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 * i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 * i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 * i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 * u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 * u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 * u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 * f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 * f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 * f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 * i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 * i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 * i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 * u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 * u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 * u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 * f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 * f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 * f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 * i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 * i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 * i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 * u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 * u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 * u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 * f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 * f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 * f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 * i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 * i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 * i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 * u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 * u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 * u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 * f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 * f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 * f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 * i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 * i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 * i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 * u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 * u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 * u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 * f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 * f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 * f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 * i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 * i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 * i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 * u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 * u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 * u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 * f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 * f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 * f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 * i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 * i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 * i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 * i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 * u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 * u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 * u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 * f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 * f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 * i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 * i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 * i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 * i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 * u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 * u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 * u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 * f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 * f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 * b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 * i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 * i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 * i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 * i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 * u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 * u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 * u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 * f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 * f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}

  (void)(s * i8);   // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * i16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * i32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * i64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * u16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * u32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * u64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * f16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * f32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s * f64); // expected-error{{cannot convert between vector and non-scalar values}}
}

void div(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b, struct S s) {
  (void)(b / b); // expected-error{{invalid operands to binary expression}}

  (void)(i8 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 / i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 / i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 / i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 / u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 / u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 / u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 / f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 / f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 / f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 / i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 / i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 / i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 / u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 / u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 / u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 / f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 / f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 / f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 / i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 / i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 / i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 / u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 / u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 / u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 / f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 / f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 / f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 / i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 / i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 / i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 / u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 / u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 / u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 / f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 / f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 / f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 / i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 / i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 / i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 / u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 / u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 / u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 / f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 / f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 / f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 / i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 / i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 / i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 / u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 / u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 / u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 / f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 / f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 / f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 / i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 / i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 / i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 / u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 / u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 / u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 / f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 / f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 / f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 / i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 / i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 / i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 / u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 / u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 / u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 / f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 / f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 / f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 / i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 / i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 / i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 / i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 / u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 / u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 / u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 / f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 / f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 / i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 / i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 / i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 / i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 / u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 / u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 / u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 / f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 / f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 / b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 / i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 / i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 / i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 / i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 / u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 / u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 / u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 / f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 / f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}

  (void)(s / i8);   // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / i16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / i32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / i64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / u16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / u32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / u64); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / f16); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / f32); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(s / f64); // expected-error{{cannot convert between vector and non-scalar values}}
}

void mod(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b, struct S s) {
  (void)(b % b); // expected-error{{invalid operands to binary expression}}

  (void)(i8 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 % i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 % i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 % i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 % u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 % u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 % u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u8 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 % i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 % i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 % i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 % u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 % u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 % u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i16 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 % i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 % i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 % i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 % u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 % u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 % u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i16 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u16 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 % i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 % i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 % i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 % u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 % u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 % u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u16 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i32 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 % i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 % i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 % i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 % u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 % u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 % u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i32 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u32 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 % i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 % i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 % i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 % u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 % u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 % u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u32 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i64 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 % i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 % i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 % i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 % u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 % u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 % u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u64 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 % i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 % i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 % i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 % u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 % u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 % u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(f16 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(f16 % i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 % i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f16 % u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(f32 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(f32 % i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 % i16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % i32); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % i64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f32 % u16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % u64); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(f32 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(f64 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(f64 % i8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 % i16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % i32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % i64); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % u8);  // expected-error{{invalid operands to binary expression}}
  (void)(f64 % u16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % u32); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(f64 % f32); // expected-error{{invalid operands to binary expression}}

  (void)(s % i8);  // expected-error{{invalid operands to binary expression}}
  (void)(s % i16); // expected-error{{invalid operands to binary expression}}
  (void)(s % i32); // expected-error{{invalid operands to binary expression}}
  (void)(s % i64); // expected-error{{invalid operands to binary expression}}
  (void)(s % u8);  // expected-error{{invalid operands to binary expression}}
  (void)(s % u16); // expected-error{{invalid operands to binary expression}}
  (void)(s % u32); // expected-error{{invalid operands to binary expression}}
  (void)(s % f16); // expected-error{{invalid operands to binary expression}}
  (void)(s % f32); // expected-error{{invalid operands to binary expression}}
}

svint8_t svi8(svint8_t a) {
  return a + 256; // expected-error{{cannot convert between scalar type 'int' and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
}

svint8_t svi8_128(svint8_t a) {
  return a + 128; // expected-warning{{implicit conversion from 'int' to 'svint8_t' (aka '__SVInt8_t') changes value from 128 to -128}}
}
