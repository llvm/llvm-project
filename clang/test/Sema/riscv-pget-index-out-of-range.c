// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <riscv_packed_simd.h>

// expected-note@*:* 2 {{candidate disabled: index must be a constant integer from 0 to 3}}
// expected-note@*:* {{candidate disabled: index must be a constant integer from 0 to 1}}
// expected-note@*:* {{candidate disabled: index must be a constant integer from 0 to 7}}
int8_t test_pget_nonconstant(int8x4_t v, unsigned idx) {
  // expected-error@+1 {{no matching function for call to '__riscv_pget_i8x4_i8'}}
  return __riscv_pget_i8x4_i8(v, idx);
}

int16_t test_pget_i16x2_out_of_range(int16x2_t v) {
  // expected-error@+1 {{no matching function for call to '__riscv_pget_i16x2_i16'}}
  return __riscv_pget_i16x2_i16(v, 2);
}

uint8_t test_pget_u8x4_out_of_range(uint8x4_t v) {
  // expected-error@+1 {{no matching function for call to '__riscv_pget_u8x4_u8'}}
  return __riscv_pget_u8x4_u8(v, 4);
}

uint8_t test_pget_u8x8_out_of_range(uint8x8_t v) {
  // expected-error@+1 {{no matching function for call to '__riscv_pget_u8x8_u8'}}
  return __riscv_pget_u8x8_u8(v, 8);
}
