// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify %s

#include <riscv_packed_simd.h>

int16x2_t test_psati_i16x2_nonconstant(int16x2_t v, unsigned width) {
  // expected-error@+1 {{argument to '__builtin_riscv_psati_i16x2' must be a constant integer}}
  return __riscv_psati_i16x2(v, width);
}

int16x2_t test_psati_i16x2_min_width(int16x2_t v) {
  return __riscv_psati_i16x2(v, 1);
}

int16x2_t test_psati_i16x2_max_width(int16x2_t v) {
  return __riscv_psati_i16x2(v, 16);
}

uint16x2_t test_pusati_u16x2_max_width(int16x2_t v) {
  return __riscv_pusati_u16x2(v, 15);
}

int16x2_t test_psati_i16x2_out_of_range(int16x2_t v) {
  // expected-error@+1 {{argument value 17 is outside the valid range [1, 16]}}
  return __riscv_psati_i16x2(v, 17);
}

int16x4_t test_psati_i16x4_out_of_range(int16x4_t v) {
  // expected-error@+1 {{argument value 0 is outside the valid range [1, 16]}}
  return __riscv_psati_i16x4(v, 0);
}

int32x2_t test_psati_i32x2_out_of_range(int32x2_t v) {
  // expected-error@+1 {{argument value 33 is outside the valid range [1, 32]}}
  return __riscv_psati_i32x2(v, 33);
}

uint16x2_t test_pusati_u16x2_out_of_range(int16x2_t v) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  return __riscv_pusati_u16x2(v, 16);
}

uint16x4_t test_pusati_u16x4_out_of_range(int16x4_t v) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  return __riscv_pusati_u16x4(v, 16);
}

uint32x2_t test_pusati_u32x2_out_of_range(int32x2_t v) {
  // expected-error@+1 {{argument value 32 is outside the valid range [0, 31]}}
  return __riscv_pusati_u32x2(v, 32);
}
