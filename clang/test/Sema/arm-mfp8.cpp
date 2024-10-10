// RUN: %clang_cc1 -fsyntax-only -verify=sve -triple aarch64-arm-none-eabi \
// RUN: -target-feature -fp8 -target-feature +sve %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>
void test_vector_sve(svmfloat8_t a, svuint8_t c) {
  a + c;  // sve-error {{cannot convert between vector type 'svuint8_t' (aka '__SVUint8_t') and vector type 'svmfloat8_t' (aka '__SVMfloat8_t') as implicit conversion would cause truncation}}
  a - c;  // sve-error {{cannot convert between vector type 'svuint8_t' (aka '__SVUint8_t') and vector type 'svmfloat8_t' (aka '__SVMfloat8_t') as implicit conversion would cause truncation}}
  a * c;  // sve-error {{cannot convert between vector type 'svuint8_t' (aka '__SVUint8_t') and vector type 'svmfloat8_t' (aka '__SVMfloat8_t') as implicit conversion would cause truncation}}
  a / c;  // sve-error {{cannot convert between vector type 'svuint8_t' (aka '__SVUint8_t') and vector type 'svmfloat8_t' (aka '__SVMfloat8_t') as implicit conversion would cause truncation}}
}

