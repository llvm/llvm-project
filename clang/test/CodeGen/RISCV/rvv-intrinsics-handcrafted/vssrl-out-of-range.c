// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vuint32m1_t test_vssrl_vv_u32m1(vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1(op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1(vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1(op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vv_u32m1_m(vbool32_t mask, vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1_m(mask, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1_m(vbool32_t mask, vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1_m(mask, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vv_u32m1_tu(vuint32m1_t maskedoff, vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1_tu(maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1_tu(vuint32m1_t maskedoff, vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1_tu(maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vv_u32m1_tum(
  vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1_tum(mask, maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1_tum(vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1_tum(mask, maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vv_u32m1_tumu(vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1_tumu(vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vv_u32m1_mu(vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, vuint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vv_u32m1_mu(mask, maskedoff, op1, op2, 5, vl);
}

vuint32m1_t test_vssrl_vx_u32m1_mu(vbool32_t mask, vuint32m1_t maskedoff, vuint32m1_t op1, uint32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vssrl_vx_u32m1_mu(mask, maskedoff, op1, op2, 5, vl);
}
