// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +experimental-zvfh \
// RUN:   -target-feature +experimental-zvkned -fsyntax-only -verify %s

#include <riscv_vector.h>

vuint32m1_t test_vaeskf2_vi_u32m1(vuint32m1_t vd, vuint32m1_t vs2, size_t vl) {
// expected-error@+1 {{argument value 33 is outside the valid range [0, 31]}}
  return __riscv_vaeskf2_vi_u32m1(vd, vs2, 33, vl);
}

vuint32m1_t test_vaeskf2_vi_u32m1_tu(vuint32m1_t merge, vuint32m1_t vs2, size_t vl) {
// expected-error@+1 {{argument value 33 is outside the valid range [0, 31]}}
  return __riscv_vaeskf2_vi_u32m1_tu(merge, vs2, 33, vl);
}
