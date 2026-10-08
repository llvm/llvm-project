// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <riscv_packed_simd.h>

// The __riscv_pst_* intrinsics take a pointer to the element type; passing a
// pointer to an unrelated type is ill-formed.

void test_pst_i8x4_ok(int8_t *p, int8x4_t v) { __riscv_pst_i8x4(p, v); }

void test_pst_i8x4_void_ptr(void *p, int8x4_t v) {
  __riscv_pst_i8x4(p, v);
}

void test_pst_i8x4_array(int8x4_t v) {
  int8_t a[4];
  __riscv_pst_i8x4(a, v);
}

void test_pst_i8x4_wrong_pointer_type(float *p, int8x4_t v) {
  // expected-error@+1 {{incompatible pointer types passing 'float *' to parameter of type 'int8_t *' (aka 'signed char *')}}
  __riscv_pst_i8x4(p, v);
}

void test_pst_u16x2_wrong_pointer_type(uint32_t *p, uint16x2_t v) {
  // expected-error@+1 {{incompatible pointer types passing 'uint32_t *' (aka 'unsigned int *') to parameter of type 'uint16_t *' (aka 'unsigned short *')}}
  __riscv_pst_u16x2(p, v);
}

void test_pst_i32x2_wrong_pointer_type(int8x4_t *p, int32x2_t v) {
  // expected-error@+1 {{incompatible pointer types passing 'int8x4_t *' to parameter of type 'int32_t *' (aka 'int *')}}
  __riscv_pst_i32x2(p, v);
}

void test_pst_i16x4_const_discards_qualifiers(const int16_t *p,
                                              int16x4_t v) {
  // expected-warning@+1 {{passing 'const int16_t *' (aka 'const short *') to parameter of type 'int16_t *' (aka 'short *') discards qualifiers}}
  __riscv_pst_i16x4(p, v);
}
