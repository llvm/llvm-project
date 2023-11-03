// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvknha %s -fsyntax-only -verify

#include <riscv_vector.h>

void test_zvk_features() {
  // zvknhb
  __riscv_vsha2ch_vv_u64m1(); // expected-error {{call to undeclared function '__riscv_vsha2ch_vv_u64m1'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsha2cl_vv_u64m1(); // expected-error {{call to undeclared function '__riscv_vsha2cl_vv_u64m1'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsha2ms_vv_u64m1(); // expected-error {{call to undeclared function '__riscv_vsha2ms_vv_u64m1'; ISO C99 and later do not support implicit function declarations}}
}
