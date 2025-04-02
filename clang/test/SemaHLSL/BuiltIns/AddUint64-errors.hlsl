// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

uint2 test_too_few_arg() {
  return __builtin_hlsl_adduint64();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

uint4 test_too_many_arg(uint4 a) {
  return __builtin_hlsl_adduint64(a, a, a);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

uint2 test_mismatched_arg_types(uint2 a, uint4 b) {
  return __builtin_hlsl_adduint64(a, b);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_adduint64' must have the same type}}
}

uint2 test_bad_num_arg_elements(uint3 a, uint3 b) {
  return __builtin_hlsl_adduint64(a, b);
  // expected-error@-1 {{incorrect number of bits in vector operand (expected a multiple of 64 bits, have 96)}}
}

uint2 test_scalar_arg_type(uint a) {
  return __builtin_hlsl_adduint64(a, a);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_adduint64' must be vectors}}
}

uint2 test_uint64_args(uint16_t2 a) {
  return __builtin_hlsl_adduint64(a, a);
  // expected-error@-1 {{incorrect number of bits in integer (expected 32 bits, have 16)}}
}

uint2 test_signed_integer_args(int2 a, int2 b) {
  return __builtin_hlsl_adduint64(a, b);
// expected-error@-1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int' (vector of 2 'unsigned int' values)}}
}

struct S {
  uint2 a;
};

uint2 test_incorrect_arg_type(S a) {
  return __builtin_hlsl_adduint64(a, a);
  // expected-error@-1 {{passing 'S' to parameter of incompatible type 'unsigned int'}}
}

