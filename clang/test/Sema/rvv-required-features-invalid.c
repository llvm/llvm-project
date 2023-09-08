// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv32 -target-feature +v %s -fsyntax-only -verify

#include <riscv_vector.h>
#include <sifive_vector.h>

vint8m1_t test_vloxei64_v_i8m1(const int8_t *base, vuint64m8_t bindex, size_t vl) {
  return __riscv_vloxei64(base, bindex, vl); // expected-error {{call to undeclared function '__riscv_vloxei64'}} expected-error {{returning 'int' from a function with incompatible result type 'vint8m1_t'}}
}

void test_vsoxei64_v_i8m1(int8_t *base, vuint64m8_t bindex, vint8m1_t value, size_t vl) {
  __riscv_vsoxei64(base, bindex, value, vl); // expected-error {{call to undeclared function '__riscv_vsoxei64'}}
}

void test_xsfvcp_sf_vc_x_se_u64m1(uint64_t rs1, size_t vl) {
  __riscv_sf_vc_x_se_u64m1(1, 1, 1, rs1, vl); // expected-error {{call to undeclared function '__riscv_sf_vc_x_se_u64m1'}}
}
