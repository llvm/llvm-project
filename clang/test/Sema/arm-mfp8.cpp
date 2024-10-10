// RUN: %clang_cc1 -fsyntax-only -verify=sve -triple aarch64-arm-none-eabi \
// RUN: -target-feature -fp8 -target-feature +sve %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>
void test_vector_sve(svmfloat8_t a, svuint8_t c) {
  a + c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a - c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a * c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a / c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
}

