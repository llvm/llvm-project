// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvbitmanip -fsyntax-only -verify=xcv %s
// RUN: %clang_cc1 -triple riscv32 -DNO_XCVBITMANIP -fsyntax-only -verify=noxcv %s

#include <stdint.h>

#ifndef NO_XCVBITMANIP
void test_bitmanip_bitrev_range(uint32_t a) {
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 31, 3);
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, -1, 0); // xcv-error {{argument value 255 is outside the valid range [0, 31]}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 32, 0); // xcv-error {{argument value 32 is outside the valid range [0, 31]}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 0, -1); // xcv-error {{argument value 255 is outside the valid range [0, 3]}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 0, 4);  // xcv-error {{argument value 4 is outside the valid range [0, 3]}}
}

void test_bitmanip_bitrev_constant(uint32_t a, uint8_t b) {
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, b, 0); // xcv-error {{argument to '__builtin_riscv_cv_bitmanip_bitrev' must be a constant integer}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 0, b); // xcv-error {{argument to '__builtin_riscv_cv_bitmanip_bitrev' must be a constant integer}}
}
#else
void test_bitmanip_requires_xcvbitmanip(uint32_t a, uint16_t b, uint32_t k) {
  (void)__builtin_riscv_cv_bitmanip_extract(a, b);    // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_extractu(a, b);   // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_insert(a, b, k);  // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_bclr(a, b);       // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_bset(a, b);       // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_ff1(a);           // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_fl1(a);           // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_clb(a);           // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_cnt(a);           // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_ror(a, k);        // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
  (void)__builtin_riscv_cv_bitmanip_bitrev(a, 31, 3); // noxcv-error {{builtin requires at least one of the following extensions: xcvbitmanip}}
}
#endif
