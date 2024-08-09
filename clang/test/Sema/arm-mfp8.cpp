// RUN: %clang_cc1 -fsyntax-only -verify=scalar,neon,sve -triple aarch64-arm-none-eabi \
// RUN: -target-feature -fp8 -target-feature +neon  -target-feature +sve %s

// REQUIRES: aarch64-registered-target
__mfp8 test_static_cast_from_char(char in) {
  return static_cast<__mfp8>(in); // scalar-error {{static_cast from 'char' to '__mfp8' is not allowed}}
}

char test_static_cast_to_char(__mfp8 in) {
  return static_cast<char>(in); // scalar-error {{static_cast from '__mfp8' to 'char' is not allowed}}
}
void test(bool b) {
  __mfp8 mfp8;

  mfp8 + mfp8;  // scalar-error {{invalid operands to binary expression ('__mfp8' and '__mfp8')}}
  mfp8 - mfp8;  // scalar-error {{invalid operands to binary expression ('__mfp8' and '__mfp8')}}
  mfp8 * mfp8;  // scalar-error {{invalid operands to binary expression ('__mfp8' and '__mfp8')}}
  mfp8 / mfp8;  // scalar-error {{invalid operands to binary expression ('__mfp8' and '__mfp8')}}
  ++mfp8;       // scalar-error {{cannot increment value of type '__mfp8'}}
  --mfp8;       // scalar-error {{cannot decrement value of type '__mfp8'}}

  char u8;

  mfp8 + u8;   // scalar-error {{invalid operands to binary expression ('__mfp8' and 'char')}}
  u8 + mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__mfp8')}}
  mfp8 - u8;   // scalar-error {{invalid operands to binary expression ('__mfp8' and 'char')}}
  u8 - mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__mfp8')}}
  mfp8 * u8;   // scalar-error {{invalid operands to binary expression ('__mfp8' and 'char')}}
  u8 * mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__mfp8')}}
  mfp8 / u8;   // scalar-error {{invalid operands to binary expression ('__mfp8' and 'char')}}
  u8 / mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__mfp8')}}
  mfp8 = u8;   // scalar-error {{assigning to '__mfp8' from incompatible type 'char'}}
  u8 = mfp8;   // scalar-error {{assigning to 'char' from incompatible type '__mfp8'}}
  mfp8 + (b ? u8 : mfp8);  // scalar-error {{incompatible operand types ('char' and '__mfp8')}}
}

#include <arm_neon.h>

void test_vector(mfloat8x8_t a, mfloat8x8_t b, uint8x8_t c) {
  a + b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a - b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a * b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a / b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}

  a + c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a - c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a * c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a / c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  c + b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c - b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c * b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c / b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
}

#include <arm_sve.h>
void test_vector_sve(svmfloat8_t a, svuint8_t c) {
  a + c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a - c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a * c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
  a / c;  // sve-error {{cannot convert between vector and non-scalar values ('svmfloat8_t' (aka '__SVMfloat8_t') and 'svuint8_t' (aka '__SVUint8_t'))}}
}

