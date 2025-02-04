// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>


vfloat32mf2_t test_exp_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_exp(v);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

vfloat32mf2_t test_exp2_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_exp2(v);
  // expected-error@-1 {{1st argument must be a floating point type}}
}
