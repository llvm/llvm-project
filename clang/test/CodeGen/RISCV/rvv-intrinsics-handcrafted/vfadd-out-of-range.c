// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vfloat32m1_t test_vfadd_vv_f32m1(vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm(op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1(vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm(op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vv_f32m1_m(vbool32_t mask, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm_m(mask, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1_m(vbool32_t mask, vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm_m(mask, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vv_f32m1_tu(vfloat32m1_t maskedoff, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm_tu(maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1_tu(vfloat32m1_t maskedoff, vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm_tu(maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vv_f32m1_tum(
  vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm_tum(mask, maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1_tum(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm_tum(mask, maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vv_f32m1_tumu(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1_tumu(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vv_f32m1_mu(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, vfloat32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vv_f32m1_rm_mu(mask, maskedoff, op1, op2, 5, vl);
}

vfloat32m1_t test_vfadd_vf_f32m1_mu(vbool32_t mask, vfloat32m1_t maskedoff, vfloat32m1_t op1, float op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfadd_vf_f32m1_rm_mu(mask, maskedoff, op1, op2, 5, vl);
}
