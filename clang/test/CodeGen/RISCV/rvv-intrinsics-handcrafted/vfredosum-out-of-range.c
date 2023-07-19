// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vfloat32m1_t test_vfredosum_vs_f32m1_f32m1_rm(vfloat32m1_t vector, vfloat32m1_t scalar, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfredosum_vs_f32m1_f32m1_rm(vector, scalar, 5, vl);
}

vfloat32m1_t test_vfredosum_vs_f32m1_f32m1_rm_m(vbool32_t mask, vfloat32m1_t vector, vfloat32m1_t scalar, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfredosum_vs_f32m1_f32m1_rm_m(mask, vector, scalar, 5, vl);
}

vfloat32m1_t test_vfredosum_vs_f32m1_f32m1_rm_tu(vfloat32m1_t maskedoff, vfloat32m1_t vector, vfloat32m1_t scalar, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfredosum_vs_f32m1_f32m1_rm_tu(maskedoff, vector, scalar, 5, vl);
}

vfloat32m1_t test_vfredosum_vs_f32m1_f32m1_rm_tum(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t vector, vfloat32m1_t scalar, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfredosum_vs_f32m1_f32m1_rm_tum(mask, maskedoff, vector, scalar, 5, vl);
}
