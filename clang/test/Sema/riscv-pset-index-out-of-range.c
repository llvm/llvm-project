// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -fsyntax-only -verify -verify-ignore-unexpected=note %s

#include <riscv_packed_simd.h>

// expected-note@*:* {{candidate disabled: index must be a constant integer from 0 to 3}}
// expected-note@*:* {{candidate disabled: index must be a constant integer from 0 to 1}}
// expected-note@*:* {{candidate disabled: index must be a constant integer from 0 to 7}}
int16x2_t test_pset_nonconstant(int16x2_t v, int16_t e, unsigned idx) {
  // expected-error@+1 {{no matching function for call to '__riscv_pset_i16_i16x2'}}
  return __riscv_pset_i16_i16x2(v, e, idx);
}

int16x2_t test_pset_i16_i16x2_ok(int16x2_t v, int16_t e) {
  return __riscv_pset_i16_i16x2(v, e, 1);
}

int16x2_t test_pset_i16_i16x2_out_of_range(int16x2_t v, int16_t e) {
  // expected-error@+1 {{no matching function for call to '__riscv_pset_i16_i16x2'}}
  return __riscv_pset_i16_i16x2(v, e, 2);
}

uint8x4_t test_pset_u8_u8x4_out_of_range(uint8x4_t v, uint8_t e) {
  // expected-error@+1 {{no matching function for call to '__riscv_pset_u8_u8x4'}}
  return __riscv_pset_u8_u8x4(v, e, 4);
}

uint8x8_t test_pset_u8_u8x8_out_of_range(uint8x8_t v, uint8_t e) {
  // expected-error@+1 {{no matching function for call to '__riscv_pset_u8_u8x8'}}
  return __riscv_pset_u8_u8x8(v, e, 8);
}
