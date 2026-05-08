// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -flax-vector-conversions=none -target-feature +future-vector \
// RUN:   -target-feature +vsx -target-feature +isa-future-instructions -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -flax-vector-conversions=none -target-feature +future-vector \
// RUN:   -target-feature +vsx -target-feature +isa-future-instructions -verify %s

// AI Assissted.

#include <altivec.h>

vector unsigned char vuca, vucb;
vector signed int vsia;

void test_invalid_params(void) {
  vector unsigned char res;

  // Test invalid parameter types
  res = vec_uncompresshn(vsia, vucb); // expected-error {{passing '__vector int' (vector of 4 'int' values) to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}} expected-note@altivec.h:* {{passing argument to parameter '__a' here}}
  res = vec_uncompressln(vuca, vsia); // expected-error {{passing '__vector int' (vector of 4 'int' values) to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}} expected-note@altivec.h:* {{passing argument to parameter '__b' here}}
  res = vec_unpack_hsn_to_byte(vsia); // expected-error {{passing '__vector int' (vector of 4 'int' values) to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}} expected-note@altivec.h:* {{passing argument to parameter '__a' here}}
}

void test_invalid_immediates(void) {
  vector unsigned char res;

  // Test out-of-range immediate values for vec_unpack_int4_to_bf16 (valid range: 0-3)
  res = vec_unpack_int4_to_bf16(vuca, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res = vec_unpack_int4_to_bf16(vuca, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}

  // Test out-of-range immediate values for vec_unpack_int8_to_bf16 (valid range: 0-1)
  res = vec_unpack_int8_to_bf16(vuca, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res = vec_unpack_int8_to_bf16(vuca, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}

  // Test out-of-range immediate values for vec_unpack_int4_to_fp32 (valid range: 0-7)
  res = vec_unpack_int4_to_fp32(vuca, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res = vec_unpack_int4_to_fp32(vuca, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}

  // Test out-of-range immediate values for vec_unpack_int8_to_fp32 (valid range: 0-3)
  res = vec_unpack_int8_to_fp32(vuca, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res = vec_unpack_int8_to_fp32(vuca, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
}

void test_non_constant_immediates(void) {
  vector unsigned char res;
  unsigned int imm = 1;

  // Test non-constant immediate values
  res = vec_unpack_int4_to_bf16(vuca, imm); // expected-error {{argument to '__builtin_altivec_vupkint4tobf16' must be a constant integer}}
  res = vec_unpack_int8_to_bf16(vuca, imm); // expected-error {{argument to '__builtin_altivec_vupkint8tobf16' must be a constant integer}}
  res = vec_unpack_int4_to_fp32(vuca, imm); // expected-error {{argument to '__builtin_altivec_vupkint4tofp32' must be a constant integer}}
  res = vec_unpack_int8_to_fp32(vuca, imm); // expected-error {{argument to '__builtin_altivec_vupkint8tofp32' must be a constant integer}}
}
