// RUN: %clang_cc1 -verify -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

void eq(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
        svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
        svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
        svbool_t b) {
  (void)(i8 == b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 == i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 == i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 == i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 == u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 == u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 == u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 == f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 == f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 == f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 == b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 == i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 == i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 == i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 == u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 == u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 == u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 == f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 == f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 == f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 == b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 == i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 == i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 == i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 == u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 == u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 == u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 == f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 == f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 == f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 == b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 == i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 == i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 == i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 == u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 == u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 == u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 == f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 == f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 == f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 == b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 == i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 == i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 == i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 == u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 == u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 == u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 == f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 == f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 == f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 == b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 == i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 == i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 == i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 == u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 == u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 == u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 == f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 == f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 == f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 == b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 == i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 == i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 == i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 == u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 == u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 == u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 == f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 == f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 == f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 == b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 == i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 == i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 == i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 == u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 == u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 == u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 == f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 == f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 == f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 == b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 == i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 == i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 == i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 == i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 == u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 == u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 == u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 == f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 == f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 == b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 == i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 == i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 == i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 == i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 == u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 == u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 == u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 == f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 == f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 == b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 == i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 == i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 == i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 == i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 == u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 == u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 == u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 == f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 == f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}

void neq(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b) {
  (void)(i8 != b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 != i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 != i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 != i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 != u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 != u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 != u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 != f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 != f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 != f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 != b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 != i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 != i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 != i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 != u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 != u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 != u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 != f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 != f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 != f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 != b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 != i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 != i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 != i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 != u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 != u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 != u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 != f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 != f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 != f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 != b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 != i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 != i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 != i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 != u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 != u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 != u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 != f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 != f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 != f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 != b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 != i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 != i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 != i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 != u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 != u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 != u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 != f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 != f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 != f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 != b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 != i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 != i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 != i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 != u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 != u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 != u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 != f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 != f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 != f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 != b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 != i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 != i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 != i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 != u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 != u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 != u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 != f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 != f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 != f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 != b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 != i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 != i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 != i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 != u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 != u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 != u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 != f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 != f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 != f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 != b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 != i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 != i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 != i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 != i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 != u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 != u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 != u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 != f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 != f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 != b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 != i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 != i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 != i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 != i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 != u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 != u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 != u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 != f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 != f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 != b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 != i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 != i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 != i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 != i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 != u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 != u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 != u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 != f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 != f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}

void lt(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
        svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
        svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
        svbool_t b) {
  (void)(i8 < b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 < i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 < i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 < i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 < u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 < u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 < u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 < f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 < f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 < f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 < b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 < i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 < i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 < i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 < u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 < u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 < u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 < f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 < f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 < f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 < b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 < i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 < i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 < i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 < u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 < u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 < u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 < f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 < f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 < f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 < b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 < i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 < i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 < i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 < u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 < u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 < u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 < f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 < f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 < f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 < b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 < i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 < i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 < i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 < u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 < u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 < u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 < f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 < f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 < f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 < b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 < i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 < i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 < i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 < u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 < u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 < u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 < f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 < f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 < f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 < b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 < i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 < i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 < i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 < u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 < u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 < u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 < f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 < f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 < f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 < b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 < i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 < i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 < i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 < u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 < u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 < u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 < f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 < f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 < f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 < b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 < i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 < i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 < i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 < i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 < u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 < u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 < u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 < f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 < f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 < b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 < i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 < i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 < i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 < i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 < u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 < u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 < u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 < f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 < f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 < b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 < i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 < i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 < i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 < i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 < u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 < u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 < u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 < f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 < f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}

void leq(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b) {
  (void)(i8 <= b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 <= i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 <= i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 <= i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 <= u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 <= u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 <= u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 <= f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 <= f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 <= f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 <= b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 <= i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 <= i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 <= i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 <= u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 <= u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 <= u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 <= f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 <= f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 <= f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 <= b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 <= i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 <= i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 <= u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 <= u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 <= f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 <= f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 <= f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 <= b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 <= i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 <= i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 <= u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 <= u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 <= f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 <= f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 <= f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 <= b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 <= i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 <= i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 <= u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 <= u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 <= f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 <= f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 <= f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 <= b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 <= i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 <= i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 <= u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 <= u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 <= f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 <= f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 <= f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 <= b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 <= i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 <= i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 <= u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 <= u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 <= f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 <= f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 <= f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 <= b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 <= i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 <= i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 <= u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 <= u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 <= f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 <= f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 <= f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 <= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 <= i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 <= i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 <= i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 <= u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 <= u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 <= f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 <= f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 <= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 <= i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 <= i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 <= i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 <= u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 <= u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 <= f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 <= f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 <= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 <= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 <= i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 <= i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 <= i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 <= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 <= u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 <= u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 <= f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 <= f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}

void gt(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
        svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
        svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
        svbool_t b) {
  (void)(i8 > b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 > i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 > i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 > i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 > u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 > u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 > u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 > f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 > f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 > f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 > b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 > i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 > i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 > i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 > u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 > u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 > u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 > f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 > f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 > f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 > b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 > i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 > i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 > i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 > u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 > u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 > u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 > f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 > f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 > f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 > b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 > i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 > i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 > i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 > u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 > u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 > u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 > f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 > f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 > f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 > b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 > i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 > i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 > i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 > u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 > u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 > u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 > f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 > f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 > f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 > b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 > i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 > i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 > i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 > u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 > u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 > u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 > f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 > f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 > f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 > b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 > i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 > i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 > i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 > u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 > u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 > u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 > f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 > f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 > f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 > b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 > i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 > i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 > i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 > u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 > u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 > u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 > f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 > f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 > f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 > b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 > i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 > i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 > i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 > i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 > u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 > u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 > u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 > f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 > f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 > b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 > i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 > i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 > i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 > i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 > u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 > u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 > u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 > f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 > f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 > b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 > i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 > i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 > i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 > i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 > u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 > u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 > u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 > f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 > f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}

void geq(svint8_t i8, svint16_t i16, svint32_t i32, svint64_t i64,
         svuint8_t u8, svuint16_t u16, svuint32_t u32, svuint64_t u64,
         svfloat16_t f16, svfloat32_t f32, svfloat64_t f64,
         svbool_t b) {
  (void)(i8 >= b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svint8_t' (aka '__SVInt8_t') as implicit conversion would cause truncation}}
  (void)(i8 >= i16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i8 >= i32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i8 >= i64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i8 >= u16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i8 >= u32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i8 >= u64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i8 >= f16); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i8 >= f32); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i8 >= f64); // expected-error{{vector operands do not have the same number of elements ('svint8_t' (aka '__SVInt8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u8 >= b);   // expected-error{{cannot convert between vector type 'svbool_t' (aka '__SVBool_t') and vector type 'svuint8_t' (aka '__SVUint8_t') as implicit conversion would cause truncation}}
  (void)(u8 >= i16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u8 >= i32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u8 >= i64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u8 >= u16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u8 >= u32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u8 >= u64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u8 >= f16); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u8 >= f32); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u8 >= f64); // expected-error{{vector operands do not have the same number of elements ('svuint8_t' (aka '__SVUint8_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i16 >= b);   // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i16 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i16 >= i32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i16 >= i64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i16 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i16 >= u32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i16 >= u64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i16 >= f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svint16_t' (aka '__SVInt16_t') as implicit conversion would cause truncation}}
  (void)(i16 >= f32); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i16 >= f64); // expected-error{{vector operands do not have the same number of elements ('svint16_t' (aka '__SVInt16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u16 >= b);   // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u16 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u16 >= i32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u16 >= i64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u16 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u16 >= u32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u16 >= u64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u16 >= f16); // expected-error{{cannot convert between vector type 'svfloat16_t' (aka '__SVFloat16_t') and vector type 'svuint16_t' (aka '__SVUint16_t') as implicit conversion would cause truncation}}
  (void)(u16 >= f32); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u16 >= f64); // expected-error{{vector operands do not have the same number of elements ('svuint16_t' (aka '__SVUint16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i32 >= b);   // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i32 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i32 >= i16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i32 >= i64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(i32 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i32 >= u16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i32 >= u64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(i32 >= f16); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i32 >= f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svint32_t' (aka '__SVInt32_t') as implicit conversion would cause truncation}}
  (void)(i32 >= f64); // expected-error{{vector operands do not have the same number of elements ('svint32_t' (aka '__SVInt32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(u32 >= b);   // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u32 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u32 >= i16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u32 >= i64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(u32 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u32 >= u16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u32 >= u64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(u32 >= f16); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u32 >= f32); // expected-error{{cannot convert between vector type 'svfloat32_t' (aka '__SVFloat32_t') and vector type 'svuint32_t' (aka '__SVUint32_t') as implicit conversion would cause truncation}}
  (void)(u32 >= f64); // expected-error{{vector operands do not have the same number of elements ('svuint32_t' (aka '__SVUint32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(i64 >= b);   // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(i64 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(i64 >= i16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(i64 >= i32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(i64 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(i64 >= u16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(i64 >= u32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(i64 >= f16); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(i64 >= f32); // expected-error{{vector operands do not have the same number of elements ('svint64_t' (aka '__SVInt64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(i64 >= f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svint64_t' (aka '__SVInt64_t') as implicit conversion would cause truncation}}

  (void)(u64 >= b);   // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(u64 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(u64 >= i16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(u64 >= i32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(u64 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(u64 >= u16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(u64 >= u32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(u64 >= f16); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(u64 >= f32); // expected-error{{vector operands do not have the same number of elements ('svuint64_t' (aka '__SVUint64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(u64 >= f64); // expected-error{{cannot convert between vector type 'svfloat64_t' (aka '__SVFloat64_t') and vector type 'svuint64_t' (aka '__SVUint64_t') as implicit conversion would cause truncation}}

  (void)(f16 >= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f16 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f16 >= i16); // expected-error{{cannot convert between vector type 'svint16_t' (aka '__SVInt16_t') and vector type 'svfloat16_t' (aka '__SVFloat16_t') as implicit conversion would cause truncation}}
  (void)(f16 >= i32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f16 >= i64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f16 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f16 >= u32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f16 >= u64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f16 >= f32); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
  (void)(f16 >= f64); // expected-error{{vector operands do not have the same number of elements ('svfloat16_t' (aka '__SVFloat16_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f32 >= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f32 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f32 >= i16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f32 >= i32); // expected-error{{cannot convert between vector type 'svint32_t' (aka '__SVInt32_t') and vector type 'svfloat32_t' (aka '__SVFloat32_t') as implicit conversion would cause truncation}}
  (void)(f32 >= i64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svint64_t' (aka '__SVInt64_t'))}}
  (void)(f32 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f32 >= u16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f32 >= u64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svuint64_t' (aka '__SVUint64_t'))}}
  (void)(f32 >= f16); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f32 >= f64); // expected-error{{vector operands do not have the same number of elements ('svfloat32_t' (aka '__SVFloat32_t') and 'svfloat64_t' (aka '__SVFloat64_t'))}}

  (void)(f64 >= b);   // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svbool_t' (aka '__SVBool_t'))}}
  (void)(f64 >= i8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint8_t' (aka '__SVInt8_t'))}}
  (void)(f64 >= i16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint16_t' (aka '__SVInt16_t'))}}
  (void)(f64 >= i32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svint32_t' (aka '__SVInt32_t'))}}
  (void)(f64 >= i64); // expected-error{{cannot convert between vector type 'svint64_t' (aka '__SVInt64_t') and vector type 'svfloat64_t' (aka '__SVFloat64_t') as implicit conversion would cause truncation}}
  (void)(f64 >= u8);  // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  (void)(f64 >= u16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint16_t' (aka '__SVUint16_t'))}}
  (void)(f64 >= u32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svuint32_t' (aka '__SVUint32_t'))}}
  (void)(f64 >= f16); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat16_t' (aka '__SVFloat16_t'))}}
  (void)(f64 >= f32); // expected-error{{vector operands do not have the same number of elements ('svfloat64_t' (aka '__SVFloat64_t') and 'svfloat32_t' (aka '__SVFloat32_t'))}}
}
