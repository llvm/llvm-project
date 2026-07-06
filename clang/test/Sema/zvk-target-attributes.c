// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zvknha %s -fsyntax-only -verify

#include <riscv_vector.h>

// expected-no-diagnostics

__attribute__((target("arch=+zvl128b")))
void test_zvk_features(vuint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl) {
  __riscv_vsha2ch_vv_u32m1(vd, vs2, vs1, vl);
}

__attribute__((target("arch=+v,+zvkn")))
vuint32m4_t testcase1(vuint32m4_t pt, vuint32m1_t rk, size_t vl)
{
  return __riscv_vaesz_vs_u32m1_u32m4(pt, rk, vl);
}

__attribute__((target("arch=+v,+zvknc")))
vuint32m4_t testcase2(vuint32m4_t pt, vuint32m1_t rk, size_t vl)
{
  return __riscv_vaesz_vs_u32m1_u32m4(pt, rk, vl);
}

__attribute__((target("arch=+v,+zvkned")))
vuint32m4_t testcase3(vuint32m4_t pt, vuint32m1_t rk, size_t vl)
{
  return __riscv_vaesz_vs_u32m1_u32m4(pt, rk, vl);
}

__attribute__((target("arch=+v,+zvkng")))
vuint32m4_t testcase4(vuint32m4_t pt, vuint32m1_t rk, size_t vl)
{
  return __riscv_vaesz_vs_u32m1_u32m4(pt, rk, vl);
}
