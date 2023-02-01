// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +experimental-zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vint8m1_t test_vget_v_index_not_constant(vint8m2_t src, int index) {
  // expected-error@+1 {{argument to '__riscv_vget_v_i8m2_i8m1' must be a constant integer}}
  return __riscv_vget_v_i8m2_i8m1(src, index);
}

vint8m1_t test_vget_v_i8m2_i8m1(vint8m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i8m2_i8m1(src, 2);
}

vint8m1_t test_vget_v_i8m4_i8m1(vint8m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i8m4_i8m1(src, 4);
}

vint8m2_t test_vget_v_i8m4_i8m2(vint8m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i8m4_i8m2(src, 2);
}

vint8m1_t test_vget_v_i8m8_i8m1(vint8m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_i8m8_i8m1(src, 8);
}

vint8m2_t test_vget_v_i8m8_i8m2(vint8m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i8m8_i8m2(src, 4);
}

vint8m4_t test_vget_v_i8m8_i8m4(vint8m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i8m8_i8m4(src, 2);
}

vint16m1_t test_vget_v_i16m2_i16m1(vint16m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i16m2_i16m1(src, 2);
}

vint16m1_t test_vget_v_i16m4_i16m1(vint16m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i16m4_i16m1(src, 4);
}

vint16m2_t test_vget_v_i16m4_i16m2(vint16m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i16m4_i16m2(src, 2);
}

vint16m1_t test_vget_v_i16m8_i16m1(vint16m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_i16m8_i16m1(src, 8);
}

vint16m2_t test_vget_v_i16m8_i16m2(vint16m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i16m8_i16m2(src, 4);
}

vint16m4_t test_vget_v_i16m8_i16m4(vint16m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i16m8_i16m4(src, 2);
}

vint32m1_t test_vget_v_i32m2_i32m1(vint32m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i32m2_i32m1(src, 2);
}

vint32m1_t test_vget_v_i32m4_i32m1(vint32m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i32m4_i32m1(src, 4);
}

vint32m2_t test_vget_v_i32m4_i32m2(vint32m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i32m4_i32m2(src, 2);
}

vint32m1_t test_vget_v_i32m8_i32m1(vint32m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_i32m8_i32m1(src, 8);
}

vint32m2_t test_vget_v_i32m8_i32m2(vint32m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i32m8_i32m2(src, 4);
}

vint32m4_t test_vget_v_i32m8_i32m4(vint32m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i32m8_i32m4(src, 2);
}

vint64m1_t test_vget_v_i64m2_i64m1(vint64m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i64m2_i64m1(src, 2);
}

vint64m1_t test_vget_v_i64m4_i64m1(vint64m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i64m4_i64m1(src, 4);
}

vint64m2_t test_vget_v_i64m4_i64m2(vint64m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i64m4_i64m2(src, 2);
}

vint64m1_t test_vget_v_i64m8_i64m1(vint64m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_i64m8_i64m1(src, 8);
}

vint64m2_t test_vget_v_i64m8_i64m2(vint64m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_i64m8_i64m2(src, 4);
}

vint64m4_t test_vget_v_i64m8_i64m4(vint64m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_i64m8_i64m4(src, 2);
}

vuint8m1_t test_vget_v_u8m2_u8m1(vuint8m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u8m2_u8m1(src, 2);
}

vuint8m1_t test_vget_v_u8m4_u8m1(vuint8m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u8m4_u8m1(src, 4);
}

vuint8m2_t test_vget_v_u8m4_u8m2(vuint8m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u8m4_u8m2(src, 2);
}

vuint8m1_t test_vget_v_u8m8_u8m1(vuint8m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_u8m8_u8m1(src, 8);
}

vuint8m2_t test_vget_v_u8m8_u8m2(vuint8m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u8m8_u8m2(src, 4);
}

vuint8m4_t test_vget_v_u8m8_u8m4(vuint8m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u8m8_u8m4(src, 2);
}

vuint16m1_t test_vget_v_u16m2_u16m1(vuint16m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u16m2_u16m1(src, 2);
}

vuint16m1_t test_vget_v_u16m4_u16m1(vuint16m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u16m4_u16m1(src, 4);
}

vuint16m2_t test_vget_v_u16m4_u16m2(vuint16m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u16m4_u16m2(src, 2);
}

vuint16m1_t test_vget_v_u16m8_u16m1(vuint16m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_u16m8_u16m1(src, 8);
}

vuint16m2_t test_vget_v_u16m8_u16m2(vuint16m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u16m8_u16m2(src, 4);
}

vuint16m4_t test_vget_v_u16m8_u16m4(vuint16m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u16m8_u16m4(src, 2);
}

vuint32m1_t test_vget_v_u32m2_u32m1(vuint32m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u32m2_u32m1(src, 2);
}

vuint32m1_t test_vget_v_u32m4_u32m1(vuint32m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u32m4_u32m1(src, 4);
}

vuint32m2_t test_vget_v_u32m4_u32m2(vuint32m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u32m4_u32m2(src, 2);
}

vuint32m1_t test_vget_v_u32m8_u32m1(vuint32m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_u32m8_u32m1(src, 8);
}

vuint32m2_t test_vget_v_u32m8_u32m2(vuint32m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u32m8_u32m2(src, 4);
}

vuint32m4_t test_vget_v_u32m8_u32m4(vuint32m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u32m8_u32m4(src, 2);
}

vuint64m1_t test_vget_v_u64m2_u64m1(vuint64m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u64m2_u64m1(src, 2);
}

vuint64m1_t test_vget_v_u64m4_u64m1(vuint64m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u64m4_u64m1(src, 4);
}

vuint64m2_t test_vget_v_u64m4_u64m2(vuint64m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u64m4_u64m2(src, 2);
}

vuint64m1_t test_vget_v_u64m8_u64m1(vuint64m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_u64m8_u64m1(src, 8);
}

vuint64m2_t test_vget_v_u64m8_u64m2(vuint64m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_u64m8_u64m2(src, 4);
}

vuint64m4_t test_vget_v_u64m8_u64m4(vuint64m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_u64m8_u64m4(src, 2);
}

vfloat32m1_t test_vget_v_f32m2_f32m1(vfloat32m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f32m2_f32m1(src, 2);
}

vfloat32m1_t test_vget_v_f32m4_f32m1(vfloat32m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f32m4_f32m1(src, 4);
}

vfloat32m2_t test_vget_v_f32m4_f32m2(vfloat32m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f32m4_f32m2(src, 2);
}

vfloat32m1_t test_vget_v_f32m8_f32m1(vfloat32m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_f32m8_f32m1(src, 8);
}

vfloat32m2_t test_vget_v_f32m8_f32m2(vfloat32m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f32m8_f32m2(src, 4);
}

vfloat32m4_t test_vget_v_f32m8_f32m4(vfloat32m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f32m8_f32m4(src, 2);
}

vfloat64m1_t test_vget_v_f64m2_f64m1(vfloat64m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f64m2_f64m1(src, 2);
}

vfloat64m1_t test_vget_v_f64m4_f64m1(vfloat64m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f64m4_f64m1(src, 4);
}

vfloat64m2_t test_vget_v_f64m4_f64m2(vfloat64m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f64m4_f64m2(src, 2);
}

vfloat64m1_t test_vget_v_f64m8_f64m1(vfloat64m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_f64m8_f64m1(src, 8);
}

vfloat64m2_t test_vget_v_f64m8_f64m2(vfloat64m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f64m8_f64m2(src, 4);
}

vfloat64m4_t test_vget_v_f64m8_f64m4(vfloat64m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f64m8_f64m4(src, 2);
}

vfloat16m1_t test_vget_v_f16m2_f16m1(vfloat16m2_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f16m2_f16m1(src, 2);
}

vfloat16m1_t test_vget_v_f16m4_f16m1(vfloat16m4_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f16m4_f16m1(src, 4);
}

vfloat16m1_t test_vget_v_f16m8_f16m1(vfloat16m8_t src) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vget_v_f16m8_f16m1(src, 8);
}

vfloat16m2_t test_vget_v_f16m4_f16m2(vfloat16m4_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f16m4_f16m2(src, 2);
}

vfloat16m2_t test_vget_v_f16m8_f16m2(vfloat16m8_t src) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vget_v_f16m8_f16m2(src, 4);
}

vfloat16m4_t test_vget_v_f16m8_f16m4(vfloat16m8_t src) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vget_v_f16m8_f16m4(src, 2);
}
