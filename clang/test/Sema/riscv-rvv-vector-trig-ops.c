// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

vfloat32mf2_t test_asin_vv_i8mf8(vfloat32mf2_t v) {

    return __builtin_elementwise_asin(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }
  
  vfloat32mf2_t test_acos_vv_i8mf8(vfloat32mf2_t v) {
  
    return __builtin_elementwise_acos(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }
  
  vfloat32mf2_t test_atan_vv_i8mf8(vfloat32mf2_t v) {
  
    return __builtin_elementwise_atan(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }

vfloat32mf2_t test_atan2_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_atan2(v, v);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

vfloat32mf2_t test_sin_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_sin(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}

vfloat32mf2_t test_cos_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_cos(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}

vfloat32mf2_t test_tan_vv_i8mf8(vfloat32mf2_t v) {

  return __builtin_elementwise_tan(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}

vfloat32mf2_t test_sinh_vv_i8mf8(vfloat32mf2_t v) {

    return __builtin_elementwise_sinh(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }
  
  vfloat32mf2_t test_cosh_vv_i8mf8(vfloat32mf2_t v) {
  
    return __builtin_elementwise_cosh(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }
  
  vfloat32mf2_t test_tanh_vv_i8mf8(vfloat32mf2_t v) {
  
    return __builtin_elementwise_tanh(v);
    // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
  }
  
