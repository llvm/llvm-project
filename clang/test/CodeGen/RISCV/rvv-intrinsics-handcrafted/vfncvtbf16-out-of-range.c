// requires: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zvfbfmin \
// RUN:   -fsyntax-only -verify %s

#include <riscv_vector.h>

vbfloat16m1_t test_vfncvtbf16_f_f_w_bf16m1_rm_m(vbool16_t mask, vfloat32m2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfncvtbf16_f_f_w_bf16m1_rm_m(mask, src, 5, vl);
}

vbfloat16m1_t test_vfncvtbf16_f_f_w_bf16m1_rm_tu(vbfloat16m1_t maskedoff, vfloat32m2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfncvtbf16_f_f_w_bf16m1_rm_tu(maskedoff, src, 5, vl);
}

vbfloat16m1_t test_vfncvtbf16_f_f_w_bf16m1_rm_tum(vbool16_t mask, vbfloat16m1_t maskedoff, vfloat32m2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfncvtbf16_f_f_w_bf16m1_rm_tum(mask, maskedoff, src, 5, vl);
}

vbfloat16m1_t test_vfncvtbf16_f_f_w_bf16m1_rm_tumu(vbool16_t mask, vbfloat16m1_t maskedoff, vfloat32m2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfncvtbf16_f_f_w_bf16m1_rm_tumu(mask, maskedoff, src, 5, vl);
}

vbfloat16m1_t test_vfncvtbf16_f_f_w_bf16m1_rm_mu(vbool16_t mask, vbfloat16m1_t maskedoff, vfloat32m2_t src, size_t vl) {
  // expected-error@+1 {{argument value 5 is outside the valid range [0, 4]}}
  return __riscv_vfncvtbf16_f_f_w_bf16m1_rm_mu(mask, maskedoff, src, 5, vl);
}
