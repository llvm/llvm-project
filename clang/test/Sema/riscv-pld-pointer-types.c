// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <riscv_packed_simd.h>

// The __riscv_pld_* intrinsics take a pointer to the element type; passing a
// pointer to an unrelated type is ill-formed.

int8x4_t test_pld_i8x4_ok(int8_t *p) {
  return __riscv_pld_i8x4(p);
}

int8x4_t test_pld_i8x4_void_ptr(void *p) {
  return __riscv_pld_i8x4(p);
}

int8x4_t test_pld_i8x4_array(void) {
  int8_t a[4] = {1, 2, 3, 4};
  return __riscv_pld_i8x4(a);
}

int8x4_t test_pld_i8x4_wrong_pointer_type(float *p) {
  // expected-error@+1 {{incompatible pointer types passing 'float *' to parameter of type 'int8_t *' (aka 'signed char *')}}
  return __riscv_pld_i8x4(p);
}

uint16x2_t test_pld_u16x2_wrong_pointer_type(uint32_t *p) {
  // expected-error@+1 {{incompatible pointer types passing 'uint32_t *' (aka 'unsigned int *') to parameter of type 'uint16_t *' (aka 'unsigned short *')}}
  return __riscv_pld_u16x2(p);
}

int32x2_t test_pld_i32x2_wrong_pointer_type(int8x4_t *p) {
  // expected-error@+1 {{incompatible pointer types passing 'int8x4_t *' to parameter of type 'int32_t *' (aka 'int *')}}
  return __riscv_pld_i32x2(p);
}

int16x4_t test_pld_i16x4_const_discards_qualifiers(const int16_t *p) {
  // expected-warning@+1 {{passing 'const int16_t *' (aka 'const short *') to parameter of type 'int16_t *' (aka 'short *') discards qualifiers}}
  return __riscv_pld_i16x4(p);
}
