// RUN: %clang_cc1 -triple aarch64 -target-feature +sve \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify
// REQUIRES: aarch64-registered-target

#include <arm_sve.h>

svfloat32_t test_pow_vv_i8mf8(svfloat32_t v) {

  return __builtin_elementwise_pow(v, v);
  // expected-error@-1 {{1st argument must be a vector, integer or floating point type}}
}
