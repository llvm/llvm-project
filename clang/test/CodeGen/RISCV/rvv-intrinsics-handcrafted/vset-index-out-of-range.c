// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +experimental-zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vint8m1_t test_vset_v_index_not_constant(vint8m2_t dest, vint8m1_t val, int index) {
  // expected-error@+1 {{argument to '__riscv_vset_v_i8m1_i8m2' must be a constant integer}}
  return __riscv_vset_v_i8m1_i8m2(dest, index, val);
}

vint8m2_t test_vset_v_i8m1_i8m2(vint8m2_t dest, vint8m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i8m1_i8m2(dest, 2, val);
}

vint8m4_t test_vset_v_i8m1_i8m4(vint8m4_t dest, vint8m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i8m1_i8m4(dest, 4, val);
}

vint8m4_t test_vset_v_i8m2_i8m4(vint8m4_t dest, vint8m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i8m2_i8m4(dest, 2, val);
}

vint8m8_t test_vset_v_i8m1_i8m8(vint8m8_t dest, vint8m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_i8m1_i8m8(dest, 8, val);
}

vint8m8_t test_vset_v_i8m2_i8m8(vint8m8_t dest, vint8m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i8m2_i8m8(dest, 4, val);
}

vint8m8_t test_vset_v_i8m4_i8m8(vint8m8_t dest, vint8m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i8m4_i8m8(dest, 2, val);
}

vint16m2_t test_vset_v_i16m1_i16m2(vint16m2_t dest, vint16m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i16m1_i16m2(dest, 2, val);
}

vint16m4_t test_vset_v_i16m1_i16m4(vint16m4_t dest, vint16m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i16m1_i16m4(dest, 4, val);
}

vint16m4_t test_vset_v_i16m2_i16m4(vint16m4_t dest, vint16m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i16m2_i16m4(dest, 2, val);
}

vint16m8_t test_vset_v_i16m1_i16m8(vint16m8_t dest, vint16m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_i16m1_i16m8(dest, 8, val);
}

vint16m8_t test_vset_v_i16m2_i16m8(vint16m8_t dest, vint16m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i16m2_i16m8(dest, 4, val);
}

vint16m8_t test_vset_v_i16m4_i16m8(vint16m8_t dest, vint16m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i16m4_i16m8(dest, 2, val);
}

vint32m2_t test_vset_v_i32m1_i32m2(vint32m2_t dest, vint32m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i32m1_i32m2(dest, 2, val);
}

vint32m4_t test_vset_v_i32m1_i32m4(vint32m4_t dest, vint32m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i32m1_i32m4(dest, 4, val);
}

vint32m4_t test_vset_v_i32m2_i32m4(vint32m4_t dest, vint32m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i32m2_i32m4(dest, 2, val);
}

vint32m8_t test_vset_v_i32m1_i32m8(vint32m8_t dest, vint32m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_i32m1_i32m8(dest, 8, val);
}

vint32m8_t test_vset_v_i32m2_i32m8(vint32m8_t dest, vint32m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i32m2_i32m8(dest, 4, val);
}

vint32m8_t test_vset_v_i32m4_i32m8(vint32m8_t dest, vint32m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i32m4_i32m8(dest, 2, val);
}

vint64m2_t test_vset_v_i64m1_i64m2(vint64m2_t dest, vint64m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i64m1_i64m2(dest, 2, val);
}

vint64m4_t test_vset_v_i64m1_i64m4(vint64m4_t dest, vint64m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i64m1_i64m4(dest, 4, val);
}

vint64m4_t test_vset_v_i64m2_i64m4(vint64m4_t dest, vint64m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i64m2_i64m4(dest, 2, val);
}

vint64m8_t test_vset_v_i64m1_i64m8(vint64m8_t dest, vint64m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_i64m1_i64m8(dest, 8, val);
}

vint64m8_t test_vset_v_i64m2_i64m8(vint64m8_t dest, vint64m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_i64m2_i64m8(dest, 4, val);
}

vint64m8_t test_vset_v_i64m4_i64m8(vint64m8_t dest, vint64m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_i64m4_i64m8(dest, 2, val);
}

vuint8m2_t test_vset_v_u8m1_u8m2(vuint8m2_t dest, vuint8m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u8m1_u8m2(dest, 2, val);
}

vuint8m4_t test_vset_v_u8m1_u8m4(vuint8m4_t dest, vuint8m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u8m1_u8m4(dest, 4, val);
}

vuint8m4_t test_vset_v_u8m2_u8m4(vuint8m4_t dest, vuint8m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u8m2_u8m4(dest, 2, val);
}

vuint8m8_t test_vset_v_u8m1_u8m8(vuint8m8_t dest, vuint8m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_u8m1_u8m8(dest, 8, val);
}

vuint8m8_t test_vset_v_u8m2_u8m8(vuint8m8_t dest, vuint8m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u8m2_u8m8(dest, 4, val);
}

vuint8m8_t test_vset_v_u8m4_u8m8(vuint8m8_t dest, vuint8m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u8m4_u8m8(dest, 2, val);
}

vuint16m2_t test_vset_v_u16m1_u16m2(vuint16m2_t dest, vuint16m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u16m1_u16m2(dest, 2, val);
}

vuint16m4_t test_vset_v_u16m1_u16m4(vuint16m4_t dest, vuint16m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u16m1_u16m4(dest, 4, val);
}

vuint16m4_t test_vset_v_u16m2_u16m4(vuint16m4_t dest, vuint16m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u16m2_u16m4(dest, 2, val);
}

vuint16m8_t test_vset_v_u16m1_u16m8(vuint16m8_t dest, vuint16m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_u16m1_u16m8(dest, 8, val);
}

vuint16m8_t test_vset_v_u16m2_u16m8(vuint16m8_t dest, vuint16m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u16m2_u16m8(dest, 4, val);
}

vuint16m8_t test_vset_v_u16m4_u16m8(vuint16m8_t dest, vuint16m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u16m4_u16m8(dest, 2, val);
}

vuint32m2_t test_vset_v_u32m1_u32m2(vuint32m2_t dest, vuint32m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u32m1_u32m2(dest, 2, val);
}

vuint32m4_t test_vset_v_u32m1_u32m4(vuint32m4_t dest, vuint32m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u32m1_u32m4(dest, 4, val);
}

vuint32m4_t test_vset_v_u32m2_u32m4(vuint32m4_t dest, vuint32m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u32m2_u32m4(dest, 2, val);
}

vuint32m8_t test_vset_v_u32m1_u32m8(vuint32m8_t dest, vuint32m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_u32m1_u32m8(dest, 8, val);
}

vuint32m8_t test_vset_v_u32m2_u32m8(vuint32m8_t dest, vuint32m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u32m2_u32m8(dest, 4, val);
}

vuint32m8_t test_vset_v_u32m4_u32m8(vuint32m8_t dest, vuint32m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u32m4_u32m8(dest, 2, val);
}

vuint64m2_t test_vset_v_u64m1_u64m2(vuint64m2_t dest, vuint64m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u64m1_u64m2(dest, 2, val);
}

vuint64m4_t test_vset_v_u64m1_u64m4(vuint64m4_t dest, vuint64m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u64m1_u64m4(dest, 4, val);
}

vuint64m4_t test_vset_v_u64m2_u64m4(vuint64m4_t dest, vuint64m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u64m2_u64m4(dest, 2, val);
}

vuint64m8_t test_vset_v_u64m1_u64m8(vuint64m8_t dest, vuint64m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_u64m1_u64m8(dest, 8, val);
}

vuint64m8_t test_vset_v_u64m2_u64m8(vuint64m8_t dest, vuint64m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_u64m2_u64m8(dest, 4, val);
}

vuint64m8_t test_vset_v_u64m4_u64m8(vuint64m8_t dest, vuint64m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_u64m4_u64m8(dest, 2, val);
}

vfloat32m2_t test_vset_v_f32m1_f32m2(vfloat32m2_t dest, vfloat32m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f32m1_f32m2(dest, 2, val);
}

vfloat32m4_t test_vset_v_f32m1_f32m4(vfloat32m4_t dest, vfloat32m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f32m1_f32m4(dest, 4, val);
}

vfloat32m4_t test_vset_v_f32m2_f32m4(vfloat32m4_t dest, vfloat32m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f32m2_f32m4(dest, 2, val);
}

vfloat32m8_t test_vset_v_f32m1_f32m8(vfloat32m8_t dest, vfloat32m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_f32m1_f32m8(dest, 8, val);
}

vfloat32m8_t test_vset_v_f32m2_f32m8(vfloat32m8_t dest, vfloat32m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f32m2_f32m8(dest, 4, val);
}

vfloat32m8_t test_vset_v_f32m4_f32m8(vfloat32m8_t dest, vfloat32m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f32m4_f32m8(dest, 2, val);
}

vfloat64m2_t test_vset_v_f64m1_f64m2(vfloat64m2_t dest, vfloat64m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f64m1_f64m2(dest, 2, val);
}

vfloat64m4_t test_vset_v_f64m1_f64m4(vfloat64m4_t dest, vfloat64m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f64m1_f64m4(dest, 4, val);
}

vfloat64m4_t test_vset_v_f64m2_f64m4(vfloat64m4_t dest, vfloat64m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f64m2_f64m4(dest, 2, val);
}

vfloat64m8_t test_vset_v_f64m1_f64m8(vfloat64m8_t dest, vfloat64m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_f64m1_f64m8(dest, 8, val);
}

vfloat64m8_t test_vset_v_f64m2_f64m8(vfloat64m8_t dest, vfloat64m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f64m2_f64m8(dest, 4, val);
}

vfloat64m8_t test_vset_v_f64m4_f64m8(vfloat64m8_t dest, vfloat64m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f64m4_f64m8(dest, 2, val);
}

vfloat16m2_t test_vset_v_f16m1_f16m2(vfloat16m2_t dest, vfloat16m1_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f16m1_f16m2(dest, 2, val);
}

vfloat16m4_t test_vset_v_f16m1_f16m4(vfloat16m4_t dest, vfloat16m1_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f16m1_f16m4(dest, 4, val);
}

vfloat16m4_t test_vset_v_f16m2_f16m4(vfloat16m4_t dest, vfloat16m2_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f16m2_f16m4(dest, 2, val);
}

vfloat16m8_t test_vset_v_f16m1_f16m8(vfloat16m8_t dest, vfloat16m1_t val) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  return __riscv_vset_v_f16m1_f16m8(dest, 8, val);
}

vfloat16m8_t test_vset_v_f16m2_f16m8(vfloat16m8_t dest, vfloat16m2_t val) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __riscv_vset_v_f16m2_f16m8(dest, 4, val);
}

vfloat16m8_t test_vset_v_f16m4_f16m8(vfloat16m8_t dest, vfloat16m4_t val) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  return __riscv_vset_v_f16m4_f16m8(dest, 2, val);
}
