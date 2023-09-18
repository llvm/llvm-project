// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +xsfvcp %s -fsyntax-only -verify

// expected-no-diagnostics

#include <riscv_vector.h>
#include <sifive_vector.h>

vint8m1_t test_vloxei64_v_i8m1(const int8_t *base, vuint64m8_t bindex, size_t vl) {
  return __riscv_vloxei64(base, bindex, vl);
}

void test_vsoxei64_v_i8m1(int8_t *base, vuint64m8_t bindex, vint8m1_t value, size_t vl) {
  __riscv_vsoxei64(base, bindex, value, vl);
}

void test_sf_vc_x_se_u64m1(uint64_t rs1, size_t vl) {
  __riscv_sf_vc_x_se_u64m1(1, 1, 1, rs1, vl);
}
