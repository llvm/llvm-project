// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

void lshift(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
            vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
            vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
            vbool1_t b) {
  (void)(b << b);

  (void)(i8 << b);
  (void)(i8 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i8 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i8 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i8 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i8 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u8 << b);
  (void)(u8 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u8 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u8 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u8 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u8 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i16 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(i16 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i16 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i16 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i16 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i16 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u16 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u16 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u16 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u16 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u16 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i32 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i32 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i32 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i32 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i32 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u32 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u32 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u32 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u32 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u32 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i64 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i64 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i64 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i64 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i64 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u64 << b);   // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u64 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u64 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u64 << 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u64 << 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(f16 << b);   // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << i8);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << i16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << i32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << i64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << u8);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << u32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << u64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << f32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << f64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << 0.f); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 << 0.);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}

  (void)(f32 << b);   // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << i8);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << i16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << i32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << i64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << u8);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << u16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << u64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << f16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << f64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 << 0.);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}

  (void)(f64 << b);   // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << i8);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << i16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << i32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << i64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << u8);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << u16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << u32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << f16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << f32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 << 0.f); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}

  (void)(b << i8);
  (void)(i16 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i8); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << i8); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << i8); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << i8); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << i8); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << i8);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << u8);
  (void)(i16 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u8); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << u8); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << u8); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << u8); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << u8); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << u8);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << i16);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << i16);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << i16);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << i16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << i16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << i16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << i16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << i16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << u16);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u16);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u16);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << u16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << u16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << u16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << u16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << u16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << i32);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << i32);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << i32);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << i32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << i32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << i32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << i32); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << i32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << u32);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u32);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u32);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 << u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << u32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << u32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << u32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << u32); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << u32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << i64);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << i64);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << i64);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << i64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << i64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << i64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << i64); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << i64);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << u64);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 << u64);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 << u64);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 << u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 << u64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << u64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << u64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << u64); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << u64);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << f16);   // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i8 << f16);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i16 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i32 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i64 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u8 << f16);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u32 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u64 << f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << f16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 << f16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f << f16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. << f16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << f32);   // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i8 << f32);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i16 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i32 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i64 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u8 << f32);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u16 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u64 << f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f16 << f32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f64 << f32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0. << f32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b << f64);   // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i8 << f64);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i16 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i32 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i64 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u8 << f64);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u16 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u32 << f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f16 << f64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 << f64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(0.f << f64); // expected-error{{used type 'float' where integer is required}}
}

void rshift(vint8m8_t i8, vint16m4_t i16, vint32m2_t i32, vint64m1_t i64,
            vuint8m8_t u8, vuint16m4_t u16, vuint32m2_t u32, vuint64m1_t u64,
            vfloat16m4_t f16, vfloat32m2_t f32, vfloat64m1_t f64,
            vbool1_t b) {
  (void)(b >> b);

  (void)(i8 >> b);
  (void)(i8 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i8 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i8 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i8 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i8 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u8 >> b);
  (void)(u8 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u8 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u8 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u8 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u8 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i16 >> b);   // expected-error{{invalid operands to binary expression ('vint16m4_t' (aka '__rvv_int16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i16 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i16 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i16 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i16 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i16 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u16 >> b);   // expected-error{{invalid operands to binary expression ('vuint16m4_t' (aka '__rvv_uint16m4_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u16 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u16 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u16 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u16 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u16 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i32 >> b);   // expected-error{{invalid operands to binary expression ('vint32m2_t' (aka '__rvv_int32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i32 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i32 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i32 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i32 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i32 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u32 >> b);   // expected-error{{invalid operands to binary expression ('vuint32m2_t' (aka '__rvv_uint32m2_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u32 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u32 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u32 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u32 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u32 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(i64 >> b);   // expected-error{{invalid operands to binary expression ('vint64m1_t' (aka '__rvv_int64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(i64 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i64 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i64 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i64 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(i64 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(u64 >> b);   // expected-error{{invalid operands to binary expression ('vuint64m1_t' (aka '__rvv_uint64m1_t') and 'vbool1_t' (aka '__rvv_bool1_t'))}}
  (void)(u64 >> i8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u8);  // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u64 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u64 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u64 >> 0.f); // expected-error{{used type 'float' where integer is required}}
  (void)(u64 >> 0.);  // expected-error{{used type 'double' where integer is required}}

  (void)(f16 >> b);   // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> i8);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> i16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> i32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> i64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> u8);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> u32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> u64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> f32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> f64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> 0.f); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f16 >> 0.);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}

  (void)(f32 >> b);   // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> i8);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> i16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> i32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> i64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> u8);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> u16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> u64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> f16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> f64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f32 >> 0.);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}

  (void)(f64 >> b);   // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> i8);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> i16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> i32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> i64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> u8);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> u16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> u32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> f16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> f32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f64 >> 0.f); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}

  (void)(b >> i8);
  (void)(i16 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> i8); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> i8); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> i8); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> i8); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> i8); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> i8);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> u8);
  (void)(i16 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u8); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> u8); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> u8); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> u8); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> u8); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> u8);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> i16);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> i16);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> i16);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> i16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> i16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> i16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> i16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> i16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> i16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> u16);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u16);  // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u16);  // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u16); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> u16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> u16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> u16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> u16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> u16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> i32);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> i32);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> i32);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> i32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> i32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> i32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> i32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> i32); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> i32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> u32);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u32);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(i64 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u32);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(u64 >> u32); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> u32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> u32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> u32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> u32); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> u32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> i64);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> i64);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> i64);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> i64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> i64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> i64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> i64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> i64); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> i64);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> u64);   // expected-error{{invalid operands to binary expression}}
  (void)(i8 >> u64);  // expected-error{{invalid operands to binary expression}}
  (void)(i16 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(i32 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(u8 >> u64);  // expected-error{{invalid operands to binary expression}}
  (void)(u16 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(u32 >> u64); // expected-error{{invalid operands to binary expression}}
  (void)(f16 >> u64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> u64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> u64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> u64); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> u64);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> f16);   // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i8 >> f16);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i16 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i32 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(i64 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u8 >> f16);  // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u32 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(u64 >> f16); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> f16); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f64 >> f16); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0.f >> f16); // expected-error{{used type 'float' where integer is required}}
  (void)(0. >> f16);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> f32);   // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i8 >> f32);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i16 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i32 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(i64 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u8 >> f32);  // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u16 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(u64 >> f32); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(f16 >> f32); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f64 >> f32); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(0. >> f32);  // expected-error{{used type 'double' where integer is required}}

  (void)(b >> f64);   // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i8 >> f64);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i16 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i32 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(i64 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u8 >> f64);  // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u16 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(u32 >> f64); // expected-error{{used type 'vfloat64m1_t' (aka '__rvv_float64m1_t') where integer is required}}
  (void)(f16 >> f64); // expected-error{{used type 'vfloat16m4_t' (aka '__rvv_float16m4_t') where integer is required}}
  (void)(f32 >> f64); // expected-error{{used type 'vfloat32m2_t' (aka '__rvv_float32m2_t') where integer is required}}
  (void)(0.f >> f64); // expected-error{{used type 'float' where integer is required}}
}
