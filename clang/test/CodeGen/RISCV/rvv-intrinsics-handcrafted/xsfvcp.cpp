// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v  \
// RUN:   -target-feature +xsfvcp \
// RUN:   -fsyntax-only -verify %s
// expected-no-diagnostics

#include <sifive_vector.h>

#define p27_26 (0b11)
#define p11_7  (0b11111)

void test_sf_vc_xv_se_u64m1(vuint64m1_t vs2, uint64_t rs1, size_t vl) {
  __riscv_sf_vc_xv_se_u64m1(p27_26, p11_7, vs2, rs1, vl);
}
