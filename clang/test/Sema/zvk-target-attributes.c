// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zvknha %s -fsyntax-only -verify

#include <riscv_vector.h>

// expected-no-diagnostics

__attribute__((target("arch=+zvl128b")))
void test_zvk_features(vuint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl) {
  __riscv_vsha2ch_vv_u32m1(vd, vs2, vs1, vl);
}
