// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -target-feature +zvksh -fsyntax-only -verify %s

#include <riscv_vector.h>

vuint32m2_t test_vsm3c_vi_u32m2(vuint32m2_t vd, vuint32m2_t vs2, size_t vl) {
// expected-error@+1 {{argument value 33 is outside the valid range [0, 31]}}
  return __riscv_vsm3c_vi_u32m2(vd, vs2, 33, vl);
}

vuint32m2_t test_vsm3c_vi_u32m2_tu(vuint32m2_t merge, vuint32m2_t vs2, size_t vl) {
// expected-error@+1 {{argument value 33 is outside the valid range [0, 31]}}
  return __riscv_vsm3c_vi_u32m2_tu(merge, vs2, 33, vl);
}
