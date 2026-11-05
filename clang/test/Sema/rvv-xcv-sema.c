// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvmac -target-feature +xcvsimd \
// RUN:   -target-feature +xcvbitmanip -fsyntax-only -verify %s

// This file tests that Sema range checks correctly reject out-of-range
// immediate arguments for XCV builtins.

#include <stdint.h>

// ===== XCVmac — shift must be in [0, 31] =====

void test_mac_range(uint32_t a, uint32_t b, uint32_t c) {
  (void)__builtin_riscv_cv_mac_muluN(a, b, 31);  // OK: max value
  (void)__builtin_riscv_cv_mac_muluN(a, b, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  (void)__builtin_riscv_cv_mac_mulsN(a, b, 0);   // OK: min value
  (void)__builtin_riscv_cv_mac_macuN(a, b, c, 31);  // OK
  (void)__builtin_riscv_cv_mac_macuN(a, b, c, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}

// ===== XCVsimd — various immediate ranges =====

void test_simd_extract_range(uint32_t a) {
  (void)__builtin_riscv_cv_simd_extract_h(a, 0);   // OK
  (void)__builtin_riscv_cv_simd_extract_h(a, 1);   // OK: max for halfword
  (void)__builtin_riscv_cv_simd_extract_h(a, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  (void)__builtin_riscv_cv_simd_extract_b(a, 3);   // OK: max for byte
  (void)__builtin_riscv_cv_simd_extract_b(a, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_simd_insert_range(uint32_t a, uint32_t b) {
  (void)__builtin_riscv_cv_simd_insert_h(a, b, 1);   // OK
  (void)__builtin_riscv_cv_simd_insert_h(a, b, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  (void)__builtin_riscv_cv_simd_insert_b(a, b, 3);   // OK
  (void)__builtin_riscv_cv_simd_insert_b(a, b, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_simd_shuffle_sci_range(uint32_t a) {
  (void)__builtin_riscv_cv_simd_shuffle_sci_h(a, 3);    // OK: max for 2-bit
  (void)__builtin_riscv_cv_simd_shuffle_sci_h(a, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)__builtin_riscv_cv_simd_shuffle_sci_b(a, 255);  // OK: max for 8-bit
  (void)__builtin_riscv_cv_simd_shuffle_sci_b(a, 256);  // expected-error {{argument value 256 is outside the valid range [0, 255]}}
}

void test_simd_cplxmul_range(uint32_t a, uint32_t b, uint32_t c) {
  (void)__builtin_riscv_cv_simd_cplxmul_r(a, b, c, 3);  // OK
  (void)__builtin_riscv_cv_simd_cplxmul_r(a, b, c, 4);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_simd_subrotmj_range(uint32_t a, uint32_t b) {
  (void)__builtin_riscv_cv_simd_subrotmj(a, b, 3);  // OK
  (void)__builtin_riscv_cv_simd_subrotmj(a, b, 4);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

// ===== XCVbitmanip — bitrev ranges =====

void test_bitmanip_bitrev_range(uint32_t a) {
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 3, 31);  // OK: max values
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 4, 0);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 0, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
}
