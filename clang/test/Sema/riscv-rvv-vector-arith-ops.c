// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

struct S {
  int x;
};

void add(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
         vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
         vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
         vbool1_t b, struct S s) {
  (void)(i8 + b); // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vint8m8_t' (aka '__rvv_int8m8_t') as implicit conversion would cause truncation}}
  (void)(i8 + i16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i8 + i32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i8 + i64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i8 + u16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i8 + u32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i8 + u64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i8 + f16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i8 + f32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i8 + f64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u8 + b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vuint8m8_t' (aka '__rvv_uint8m8_t') as implicit conversion would cause truncation}}
  (void)(u8 + i16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u8 + i32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u8 + i64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u8 + u16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u8 + u32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u8 + u64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u8 + f16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u8 + f32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u8 + f64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i16 + b);   // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i16 + i8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i16 + i32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i16 + i64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i16 + u8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i16 + u32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i16 + u64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i16 + f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vint16m4_t' (aka '__rvv_int16m4_t') as implicit conversion would cause truncation}}
  (void)(i16 + f32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i16 + f64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u16 + b);   // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u16 + i8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u16 + i32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u16 + i64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u16 + u8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u16 + u32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u16 + u64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u16 + f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vuint16m4_t' (aka '__rvv_uint16m4_t') as implicit conversion would cause truncation}}
  (void)(u16 + f32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u16 + f64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i32 + b);   // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i32 + i8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i32 + i16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i32 + i64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i32 + u8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i32 + u16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i32 + u64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i32 + f16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i32 + f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vint32m2_t' (aka '__rvv_int32m2_t') as implicit conversion would cause truncation}}
  (void)(i32 + f64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u32 + b);   // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u32 + i8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u32 + i16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u32 + i64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u32 + u8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u32 + u16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u32 + u64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u32 + f16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u32 + f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vuint32m2_t' (aka '__rvv_uint32m2_t') as implicit conversion would cause truncation}}
  (void)(u32 + f64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i64 + b);   // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i64 + i8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i64 + i16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i64 + i32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i64 + u8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i64 + u16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i64 + u32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i64 + f16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i64 + f32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i64 + f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vint64m1_t' (aka '__rvv_int64m1_t') as implicit conversion would cause truncation}}

  (void)(u64 + b);   // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u64 + i8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u64 + i16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u64 + i32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u64 + u8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u64 + u16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u64 + u32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u64 + f16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u64 + f32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u64 + f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vuint64m1_t' (aka '__rvv_uint64m1_t') as implicit conversion would cause truncation}}

  (void)(f16 + b);   // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f16 + i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f16 + i16); // expected-error{{cannot convert between vector type 'vint16m4_t' (aka '__rvv_int16m4_t') and vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') as implicit conversion would cause truncation}}
  (void)(f16 + i32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f16 + i64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f16 + u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f16 + u32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f16 + u64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f16 + f32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(f16 + f64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f32 + b);   // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f32 + i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f32 + i16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f32 + i32); // expected-error{{cannot convert between vector type 'vint32m2_t' (aka '__rvv_int32m2_t') and vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') as implicit conversion would cause truncation}}
  (void)(f32 + i64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f32 + u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f32 + u16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f32 + u64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f32 + f16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f32 + f64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f64 + b);   // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f64 + i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f64 + i16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f64 + i32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f64 + i64); // expected-error{{cannot convert between vector type 'vint64m1_t' (aka '__rvv_int64m1_t') and vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') as implicit conversion would cause truncation}}
  (void)(f64 + u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f64 + u16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f64 + u32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f64 + f16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f64 + f32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}

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

void sub(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
         vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
         vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
         vbool1_t b, struct S s) {
  (void)(i8 - b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vint8m8_t' (aka '__rvv_int8m8_t') as implicit conversion would cause truncation}}
  (void)(i8 - i16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i8 - i32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i8 - i64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i8 - u16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i8 - u32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i8 - u64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i8 - f16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i8 - f32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i8 - f64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u8 - b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vuint8m8_t' (aka '__rvv_uint8m8_t') as implicit conversion would cause truncation}}
  (void)(u8 - i16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u8 - i32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u8 - i64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u8 - u16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u8 - u32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u8 - u64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u8 - f16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u8 - f32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u8 - f64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i16 - b);   // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i16 - i8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i16 - i32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i16 - i64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i16 - u8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i16 - u32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i16 - u64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i16 - f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vint16m4_t' (aka '__rvv_int16m4_t') as implicit conversion would cause truncation}}
  (void)(i16 - f32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i16 - f64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u16 - b);   // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u16 - i8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u16 - i32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u16 - i64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u16 - u8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u16 - u32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u16 - u64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u16 - f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vuint16m4_t' (aka '__rvv_uint16m4_t') as implicit conversion would cause truncation}}
  (void)(u16 - f32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u16 - f64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i32 - b);   // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i32 - i8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i32 - i16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i32 - i64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i32 - u8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i32 - u16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i32 - u64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i32 - f16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i32 - f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vint32m2_t' (aka '__rvv_int32m2_t') as implicit conversion would cause truncation}}
  (void)(i32 - f64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u32 - b);   // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u32 - i8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u32 - i16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u32 - i64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u32 - u8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u32 - u16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u32 - u64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u32 - f16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u32 - f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vuint32m2_t' (aka '__rvv_uint32m2_t') as implicit conversion would cause truncation}}
  (void)(u32 - f64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i64 - b);   // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i64 - i8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i64 - i16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i64 - i32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i64 - u8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i64 - u16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i64 - u32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i64 - f16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i64 - f32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i64 - f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vint64m1_t' (aka '__rvv_int64m1_t') as implicit conversion would cause truncation}}

  (void)(u64 - b);   // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u64 - i8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u64 - i16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u64 - i32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u64 - u8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u64 - u16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u64 - u32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u64 - f16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u64 - f32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u64 - f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vuint64m1_t' (aka '__rvv_uint64m1_t') as implicit conversion would cause truncation}}

  (void)(f16 - b);   // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f16 - i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f16 - i16); // expected-error{{cannot convert between vector type 'vint16m4_t' (aka '__rvv_int16m4_t') and vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') as implicit conversion would cause truncation}}
  (void)(f16 - i32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f16 - i64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f16 - u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f16 - u32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f16 - u64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f16 - f32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(f16 - f64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f32 - b);   // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f32 - i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f32 - i16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f32 - i32); // expected-error{{cannot convert between vector type 'vint32m2_t' (aka '__rvv_int32m2_t') and vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') as implicit conversion would cause truncation}}
  (void)(f32 - i64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f32 - u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f32 - u16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f32 - u64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f32 - f16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f32 - f64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f64 - b);   // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f64 - i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f64 - i16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f64 - i32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f64 - i64); // expected-error{{cannot convert between vector type 'vint64m1_t' (aka '__rvv_int64m1_t') and vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') as implicit conversion would cause truncation}}
  (void)(f64 - u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f64 - u16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f64 - u32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f64 - f16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f64 - f32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}

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

void mul(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
         vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
         vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
         vbool1_t b, struct S s) {
  (void)(i8 * b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vint8m8_t' (aka '__rvv_int8m8_t') as implicit conversion would cause truncation}}
  (void)(i8 * i16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i8 * i32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i8 * i64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i8 * u16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i8 * u32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i8 * u64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i8 * f16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i8 * f32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i8 * f64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u8 * b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vuint8m8_t' (aka '__rvv_uint8m8_t') as implicit conversion would cause truncation}}
  (void)(u8 * i16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u8 * i32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u8 * i64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u8 * u16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u8 * u32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u8 * u64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u8 * f16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u8 * f32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u8 * f64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i16 * b);   // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i16 * i8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i16 * i32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i16 * i64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i16 * u8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i16 * u32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i16 * u64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i16 * f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vint16m4_t' (aka '__rvv_int16m4_t') as implicit conversion would cause truncation}}
  (void)(i16 * f32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i16 * f64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u16 * b);   // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u16 * i8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u16 * i32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u16 * i64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u16 * u8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u16 * u32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u16 * u64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u16 * f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vuint16m4_t' (aka '__rvv_uint16m4_t') as implicit conversion would cause truncation}}
  (void)(u16 * f32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u16 * f64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i32 * b);   // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i32 * i8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i32 * i16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i32 * i64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i32 * u8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i32 * u16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i32 * u64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i32 * f16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i32 * f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vint32m2_t' (aka '__rvv_int32m2_t') as implicit conversion would cause truncation}}
  (void)(i32 * f64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u32 * b);   // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u32 * i8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u32 * i16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u32 * i64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u32 * u8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u32 * u16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u32 * u64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u32 * f16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u32 * f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vuint32m2_t' (aka '__rvv_uint32m2_t') as implicit conversion would cause truncation}}
  (void)(u32 * f64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i64 * b);   // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i64 * i8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i64 * i16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i64 * i32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i64 * u8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i64 * u16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i64 * u32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i64 * f16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i64 * f32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i64 * f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vint64m1_t' (aka '__rvv_int64m1_t') as implicit conversion would cause truncation}}

  (void)(u64 * b);   // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u64 * i8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u64 * i16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u64 * i32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u64 * u8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u64 * u16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u64 * u32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u64 * f16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u64 * f32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u64 * f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vuint64m1_t' (aka '__rvv_uint64m1_t') as implicit conversion would cause truncation}}

  (void)(f16 * b);   // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f16 * i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f16 * i16); // expected-error{{cannot convert between vector type 'vint16m4_t' (aka '__rvv_int16m4_t') and vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') as implicit conversion would cause truncation}}
  (void)(f16 * i32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f16 * i64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f16 * u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f16 * u32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f16 * u64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f16 * f32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(f16 * f64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f32 * b);   // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f32 * i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f32 * i16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f32 * i32); // expected-error{{cannot convert between vector type 'vint32m2_t' (aka '__rvv_int32m2_t') and vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') as implicit conversion would cause truncation}}
  (void)(f32 * i64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f32 * u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f32 * u16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f32 * u64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f32 * f16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f32 * f64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f64 * b);   // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f64 * i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f64 * i16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f64 * i32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f64 * i64); // expected-error{{cannot convert between vector type 'vint64m1_t' (aka '__rvv_int64m1_t') and vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') as implicit conversion would cause truncation}}
  (void)(f64 * u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f64 * u16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f64 * u32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f64 * f16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f64 * f32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}

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

void div(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
         vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
         vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
         vbool1_t b, struct S s) {
  (void)(i8 / b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vint8m8_t' (aka '__rvv_int8m8_t') as implicit conversion would cause truncation}}
  (void)(i8 / i16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i8 / i32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i8 / i64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i8 / u16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i8 / u32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i8 / u64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i8 / f16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i8 / f32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i8 / f64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u8 / b);   // expected-error{{cannot convert between vector type 'vbool1_t' (aka '__rvv_bool1_t') and vector type 'vuint8m8_t' (aka '__rvv_uint8m8_t') as implicit conversion would cause truncation}}
  (void)(u8 / i16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u8 / i32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u8 / i64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u8 / u16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u8 / u32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u8 / u64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u8 / f16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u8 / f32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u8 / f64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i16 / b);   // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i16 / i8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i16 / i32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i16 / i64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i16 / u8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i16 / u32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i16 / u64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i16 / f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vint16m4_t' (aka '__rvv_int16m4_t') as implicit conversion would cause truncation}}
  (void)(i16 / f32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i16 / f64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u16 / b);   // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u16 / i8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u16 / i32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u16 / i64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u16 / u8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u16 / u32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u16 / u64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u16 / f16); // expected-error{{cannot convert between vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') and vector type 'vuint16m4_t' (aka '__rvv_uint16m4_t') as implicit conversion would cause truncation}}
  (void)(u16 / f32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u16 / f64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i32 / b);   // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i32 / i8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i32 / i16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i32 / i64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i32 / u8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i32 / u16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i32 / u64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i32 / f16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i32 / f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vint32m2_t' (aka '__rvv_int32m2_t') as implicit conversion would cause truncation}}
  (void)(i32 / f64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(u32 / b);   // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u32 / i8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u32 / i16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u32 / i64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u32 / u8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u32 / u16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u32 / u64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u32 / f16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u32 / f32); // expected-error{{cannot convert between vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') and vector type 'vuint32m2_t' (aka '__rvv_uint32m2_t') as implicit conversion would cause truncation}}
  (void)(u32 / f64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(i64 / b);   // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i64 / i8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i64 / i16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i64 / i32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i64 / u8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i64 / u16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i64 / u32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i64 / f16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(i64 / f32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(i64 / f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vint64m1_t' (aka '__rvv_int64m1_t') as implicit conversion would cause truncation}}

  (void)(u64 / b);   // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u64 / i8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u64 / i16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u64 / i32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u64 / u8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u64 / u16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u64 / u32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u64 / f16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(u64 / f32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(u64 / f64); // expected-error{{cannot convert between vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') and vector type 'vuint64m1_t' (aka '__rvv_uint64m1_t') as implicit conversion would cause truncation}}

  (void)(f16 / b);   // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f16 / i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f16 / i16); // expected-error{{cannot convert between vector type 'vint16m4_t' (aka '__rvv_int16m4_t') and vector type 'vfloat16m4_t' (aka '__rvv_float16m4_t') as implicit conversion would cause truncation}}
  (void)(f16 / i32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f16 / i64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f16 / u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f16 / u32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f16 / u64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f16 / f32); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}
  (void)(f16 / f64); // expected-error{{vector operands do not have the same number of elements ('vfloat16m4_t' (aka '__rvv_float16m4_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f32 / b);   // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f32 / i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f32 / i16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f32 / i32); // expected-error{{cannot convert between vector type 'vint32m2_t' (aka '__rvv_int32m2_t') and vector type 'vfloat32m2_t' (aka '__rvv_float32m2_t') as implicit conversion would cause truncation}}
  (void)(f32 / i64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(f32 / u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f32 / u16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f32 / u64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(f32 / f16); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f32 / f64); // expected-error{{vector operands do not have the same number of elements ('vfloat32m2_t' (aka '__rvv_float32m2_t') and 'vfloat64m1_t' (aka '__rvv_float64m1_t'))}}

  (void)(f64 / b);   // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(f64 / i8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(f64 / i16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(f64 / i32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(f64 / i64); // expected-error{{cannot convert between vector type 'vint64m1_t' (aka '__rvv_int64m1_t') and vector type 'vfloat64m1_t' (aka '__rvv_float64m1_t') as implicit conversion would cause truncation}}
  (void)(f64 / u8);  // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(f64 / u16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(f64 / u32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(f64 / f16); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat16m4_t' (aka '__rvv_float16m4_t'))}}
  (void)(f64 / f32); // expected-error{{vector operands do not have the same number of elements ('vfloat64m1_t' (aka '__rvv_float64m1_t') and 'vfloat32m2_t' (aka '__rvv_float32m2_t'))}}

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

void mod(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
         vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
         vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
         vbool1_t b, struct S s) {
  (void)(i8 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 % i16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i8 % i32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i8 % i64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i8 % u16); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i8 % u32); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i8 % u64); // expected-error{{vector operands do not have the same number of elements ('vint8m8_t' (aka '__rvv_int8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i8 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u8 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u8 % i16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u8 % i32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u8 % i64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u8 % u16); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u8 % u32); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u8 % u64); // expected-error{{vector operands do not have the same number of elements ('vuint8m8_t' (aka '__rvv_uint8m8_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u8 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i16 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 % i8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i16 % i32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i16 % i64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i16 % u8);  // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i16 % u32); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i16 % u64); // expected-error{{vector operands do not have the same number of elements ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i16 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i16 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u16 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 % i8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u16 % i32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u16 % i64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u16 % u8);  // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u16 % u32); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(u16 % u64); // expected-error{{vector operands do not have the same number of elements ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u16 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u16 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i32 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 % i8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i32 % i16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i32 % i64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(i32 % u8);  // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i32 % u16); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i32 % u64); // expected-error{{vector operands do not have the same number of elements ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(i32 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i32 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u32 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 % i8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u32 % i16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u32 % i64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vint64m1_t' (aka '__rvv_int64m1_t'))}}
  (void)(u32 % u8);  // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u32 % u16); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u32 % u64); // expected-error{{vector operands do not have the same number of elements ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vuint64m1_t' (aka '__rvv_uint64m1_t'))}}
  (void)(u32 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(u32 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(i64 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 % i8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(i64 % i16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(i64 % i32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(i64 % u8);  // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(i64 % u16); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(i64 % u32); // expected-error{{vector operands do not have the same number of elements ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
  (void)(i64 % f16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 % f32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 % f64); // expected-error{{invalid operands to binary expression}}

  (void)(u64 % b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 % i8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint8m8_t' (aka '__rvv_int8m8_t'))}}
  (void)(u64 % i16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint16m4_t' (aka '__rvv_int16m4_t'))}}
  (void)(u64 % i32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vint32m2_t' (aka '__rvv_int32m2_t'))}}
  (void)(u64 % u8);  // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint8m8_t' (aka '__rvv_uint8m8_t'))}}
  (void)(u64 % u16); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint16m4_t' (aka '__rvv_uint16m4_t'))}}
  (void)(u64 % u32); // expected-error{{vector operands do not have the same number of elements ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vuint32m2_t' (aka '__rvv_uint32m2_t'))}}
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

vint8m8_t svi8(vint8m8_t a) {
  return a + 256; // expected-error{{cannot convert between scalar type 'int' and vector type 'vint8m8_t' (aka '__rvv_int8m8_t') as implicit conversion would cause truncation}}
}

vint8m8_t svi8_128(vint8m8_t a) {
  return a + 128; // expected-warning{{implicit conversion from 'int' to 'vint8m8_t' (aka '__rvv_int8m8_t') changes value from 128 to -128}}
}
