// RUN: %clang_cc1 -triple aarch64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh  -target-feature +sve -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>


svfloat32_t test_sin_vv_i8mf8(svfloat32_t v) {
  
  return __builtin_elementwise_sin(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}

svfloat32_t test_cos_vv_i8mf8(svfloat32_t v) {

  return __builtin_elementwise_cos(v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}
