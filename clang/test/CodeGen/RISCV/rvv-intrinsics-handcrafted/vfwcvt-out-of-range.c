// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm(vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm(src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm(vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm(src, 5, vl);
}

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm_m(vbool32_t mask, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm_m(mask, src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm_m(vbool32_t mask, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm_m(mask, src, 5, vl);
}

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm_tu(vint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm_tu(maskedoff, src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm_tu(vuint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm_tu(maskedoff, src, 5, vl);
}

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm_tum(vbool32_t mask, vint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm_tum(mask, maskedoff, src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm_tum(vbool32_t mask, vuint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm_tum(mask, maskedoff, src, 5, vl);
}

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm_tumu(vbool32_t mask, vint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm_tumu(mask, maskedoff, src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm_tumu(vbool32_t mask, vuint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm_tumu(mask, maskedoff, src, 5, vl);
}

vint32m1_t test_vfwcvt_x_f_v_i32m1_rm_mu(vbool32_t mask, vint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_x_f_v_i32m1_rm_mu(mask, maskedoff, src, 5, vl);
}

vuint32m1_t test_vfwcvt_xu_f_v_u32m1_rm_mu(vbool32_t mask, vuint32m1_t maskedoff, vfloat16mf2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfwcvt_xu_f_v_u32m1_rm_mu(mask, maskedoff, src, 5, vl);
}
