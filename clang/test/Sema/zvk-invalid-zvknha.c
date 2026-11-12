// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvknha %s -fsyntax-only -verify

#include <riscv_vector.h>

void test_zvk_features(vuint64m1_t vd, vuint64m1_t vs2, vuint64m1_t vs1, size_t vl) {
  // zvknhb
  __riscv_vsha2ch_vv_u64m1(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknhb}}
  __riscv_vsha2cl_vv_u64m1(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknhb}}
  __riscv_vsha2ms_vv_u64m1(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknhb}}
}
