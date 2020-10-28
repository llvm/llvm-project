// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -fsyntax-only -ffreestanding %s -verify
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -fsyntax-only -ffreestanding -flax-vector-conversions=none %s -verify
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +simd128 -fsyntax-only -ffreestanding -x c++ %s -verify
// expected-no-diagnostics

#if defined(__wasm__) && defined(__wasm_simd128__)

#include <wasm_simd128.h>

// Test that macros are correct as well
int main() {
  v128_t v;
  v = wasm_i8x16_const(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  v = wasm_i16x8_const(0, 1, 2, 3, 4, 5, 6, 7);
  v = wasm_i32x4_const(0, 1, 2, 3);
  v = wasm_i64x2_const(0, 1);
  v = wasm_f32x4_const(0.0, 1.0, 2.0, 3.0);
  v = wasm_f64x2_const(0.0, 1.0);

  int8_t i8 = wasm_i8x16_extract_lane(v, 0);
  uint8_t u8 = wasm_u8x16_extract_lane(v, 15);
  v = wasm_i8x16_replace_lane(v, 0, 42);

  int16_t i16 = wasm_i16x8_extract_lane(v, 0);
  uint16_t u16 = wasm_u16x8_extract_lane(v, 7);
  v = wasm_i16x8_replace_lane(v, 0, 42);

  int32_t i32 = wasm_i32x4_extract_lane(v, 0);
  v = wasm_i32x4_replace_lane(v, 0, 42);

  int64_t i64 = wasm_i64x2_extract_lane(v, 0);
  v = wasm_i64x2_replace_lane(v, 0, 42);

  float f32 = wasm_f32x4_extract_lane(v, 0);
  v = wasm_f32x4_replace_lane(v, 0, 42.0);

  double f64 = wasm_f64x2_extract_lane(v, 0);
  v = wasm_f64x2_replace_lane(v, 0, 42.0);

  wasm_v8x16_shuffle(v, v, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  wasm_v16x8_shuffle(v, v, 0, 1, 2, 3, 4, 5, 6, 7);
  wasm_v32x4_shuffle(v, v, 0, 1, 2, 3);
  wasm_v64x2_shuffle(v, v, 0, 1);

  return 0;
}

#endif
