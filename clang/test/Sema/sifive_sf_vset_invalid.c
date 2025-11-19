// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +xsfmmbase -disable-O0-optnone \
// RUN:   -o - -fsyntax-only %s -verify

#include <sifive_vector.h>

void test(size_t vl) {
  __riscv_sf_vsettnt(vl, 1, 8);
  // expected-error@-1 {{argument value 8 is outside the valid range [1, 3]}}
  __riscv_sf_vsettm(vl, 8, 9);
  // expected-error@-1 {{argument value 8 is outside the valid range [0, 3]}}
  __riscv_sf_vsettn(vl, 8, 2);
  // expected-error@-1 {{argument value 8 is outside the valid range [0, 3]}}
  __riscv_sf_vsettk(vl, 0, 0);
  // expected-error@-1 {{argument value 0 is outside the valid range [1, 3]}}
}
