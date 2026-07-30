// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>


vfloat32m1_t test_vfmacc_vv_f32m1_rm(vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm(vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm(vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm(vd, rs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vv_f32m1_rm_m(vbool32_t mask, vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm_m(mask, vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm_m(vbool32_t mask, vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm_m(mask, vd, rs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vv_f32m1_rm_tu(vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm_tu(vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm_tu(vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm_tu(vd, rs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vv_f32m1_rm_tum(vbool32_t mask, vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm_tum(mask, vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm_tum(vbool32_t mask, vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm_tum(mask, vd, rs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vv_f32m1_rm_tumu(vbool32_t mask, vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm_tumu(mask, vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm_tumu(vbool32_t mask, vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm_tumu(mask, vd, rs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vv_f32m1_rm_mu(vbool32_t mask, vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vv_f32m1_rm_mu(mask, vd, vs1, vs2, 5, vl);
}

vfloat32m1_t test_vfmacc_vf_f32m1_rm_mu(vbool32_t mask, vfloat32m1_t vd, float rs1, vfloat32m1_t vs2, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfmacc_vf_f32m1_rm_mu(mask, vd, rs1, vs2, 5, vl);
}
