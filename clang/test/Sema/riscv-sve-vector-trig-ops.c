// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +experimental-zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>


vfloat32mf2_t test_sin_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_sin(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}

vfloat32mf2_t test_cos_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_cos(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}
