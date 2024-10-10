// RUN: %clang_cc1 -fsyntax-only -verify=scalar -triple aarch64-arm-none-eabi -target-feature -fp8  %s

// REQUIRES: aarch64-registered-target
__mfp8 test_static_cast_from_char(char in) {
  return static_cast<__mfp8>(in); // scalar-error {{static_cast from 'char' to '__MFloat8_t' is not allowed}}
}

char test_static_cast_to_char(__mfp8 in) {
  return static_cast<char>(in); // scalar-error {{static_cast from '__MFloat8_t' to 'char' is not allowed}}
}
void test(bool b) {
  __mfp8 mfp8;

  mfp8 + mfp8;  // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and '__MFloat8_t')}}
  mfp8 - mfp8;  // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and '__MFloat8_t')}}
  mfp8 * mfp8;  // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and '__MFloat8_t')}}
  mfp8 / mfp8;  // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and '__MFloat8_t')}}
  ++mfp8;       // scalar-error {{cannot increment value of type '__MFloat8_t'}}
  --mfp8;       // scalar-error {{cannot decrement value of type '__MFloat8_t'}}

  char u8;

  mfp8 + u8;   // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and 'char')}}
  u8 + mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__MFloat8_t')}}
  mfp8 - u8;   // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and 'char')}}
  u8 - mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__MFloat8_t')}}
  mfp8 * u8;   // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and 'char')}}
  u8 * mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__MFloat8_t')}}
  mfp8 / u8;   // scalar-error {{invalid operands to binary expression ('__MFloat8_t' and 'char')}}
  u8 / mfp8;   // scalar-error {{invalid operands to binary expression ('char' and '__MFloat8_t')}}
  mfp8 = u8;   // scalar-error {{assigning to '__MFloat8_t' from incompatible type 'char'}}
  u8 = mfp8;   // scalar-error {{assigning to 'char' from incompatible type '__MFloat8_t'}}
  mfp8 + (b ? u8 : mfp8);  // scalar-error {{incompatible operand types ('char' and '__MFloat8_t')}}
}

