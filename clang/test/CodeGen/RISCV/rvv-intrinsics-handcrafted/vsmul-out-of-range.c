// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vint32m1_t test_vsmul_vv_i32m1(vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1(op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1(vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1(op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vv_i32m1_m(vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1_m(mask, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1_m(vbool32_t mask, vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1_m(mask, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vv_i32m1_tu(vint32m1_t maskedoff, vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1_tu(maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1_tu(vint32m1_t maskedoff, vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1_tu(maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vv_i32m1_tum(
  vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1_tum(mask, maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1_tum(vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1_tum(mask, maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vv_i32m1_tumu(vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1_tumu(vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1_tumu(mask, maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vv_i32m1_mu(vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, vint32m1_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vv_i32m1_mu(mask, maskedoff, op1, op2, 5, vl);
}

vint32m1_t test_vsmul_vx_i32m1_mu(vbool32_t mask, vint32m1_t maskedoff, vint32m1_t op1, int32_t op2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 3]}}
  return __riscv_vsmul_vx_i32m1_mu(mask, maskedoff, op1, op2, 5, vl);
}
