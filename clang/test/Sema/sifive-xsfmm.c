// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +xsfmmbase -target-feature +xsfmm32a -target-feature +xsfmm32a8f \
// RUN:   -target-feature +xsfmm32a16f -target-feature +xsfmm32a32f -target-feature +xsfmm64a64f \
// RUN:   -target-feature +xsfmm32a4f -target-feature +xsfmm32a8i \
// RUN:   -target-feature +experimental-zvfofp8min -disable-O0-optnone  \
// RUN:   -fsyntax-only %s -verify
// REQUIRES: riscv-registered-target
#include <sifive_vector.h>

void test(vfloat32m8_t arg0, vuint8m8_t arg1, vfloat8e5m2m8_t arg2,
          vfloat8e4m3m8_t arg3) {
  __riscv_sf_mm_w1(4, arg0, arg0, 1, 2, 3);
  __riscv_sf_mm_w4(8, arg2, arg3, 1, 2, 3);
  __riscv_sf_mm_w4(12, arg1, arg1, 1, 2, 3);
  __riscv_sf_vtzero_t_e8w1(0, 0, 0);

  __riscv_sf_mm_w1(5, arg0, arg0, 1, 2, 3); /* expected-error {{argument should be a multiple of 4}} */
  __riscv_sf_mm_w4(7, arg2, arg3, 1, 2, 3); /* expected-error {{argument should be a multiple of 4}} */
  __riscv_sf_mm_w4(15, arg1, arg1, 1, 2, 3); /* expected-error {{argument should be a multiple of 4}} */
  __riscv_sf_mm_w1(16, arg0, arg0, 1, 2, 3); /* expected-error {{argument value 16 is outside the valid range [0, 15]}} */
  __riscv_sf_mm_w4(20, arg2, arg3, 1, 2, 3); /* expected-error {{argument value 20 is outside the valid range [0, 15]}} */
  __riscv_sf_mm_w4(24, arg1, arg1, 1, 2, 3); /* expected-error {{argument value 24 is outside the valid range [0, 15]}} */
  __riscv_sf_vtzero_t_e8w1(18, 0, 0); /* expected-error {{argument value 18 is outside the valid range [0, 15]}} */
  __riscv_sf_vtzero_t_e16w1(3, 0, 0); /* expected-error {{argument should be a multiple of 2}} */
  __riscv_sf_vtzero_t_e16w2(3, 0, 0); /* expected-error {{argument should be a multiple of 4}} */
  __riscv_sf_vtzero_t_e32w1(5, 0, 0); /* expected-error {{argument should be a multiple of 4}} */
  __riscv_sf_vtzero_t_e32w2(5, 0, 0); /* expected-error {{argument should be a multiple of 2}} */
  __riscv_sf_vtzero_t(5, 0, 0, 7, 1); /* expected-error {{argument value 7 is outside the valid range [3, 6]}} */
  __riscv_sf_vtzero_t(5, 0, 0, 2, 1); /* expected-error {{argument value 2 is outside the valid range [3, 6]}} */
  __riscv_sf_vtzero_t(5, 0, 0, 6, 3); /* expected-error {{RISC-V XSfmm twiden must be 1, 2 or 4}} */
  __riscv_sf_vtzero_t(5, 0, 0, 6, 5); /* expected-error {{RISC-V XSfmm twiden must be 1, 2 or 4}} */
}
