// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

double test_too_few_arg() {
  return __builtin_hlsl_asdouble();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

double test_too_few_arg_1(uint p0) {
  return __builtin_hlsl_asdouble(p0);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

double test_too_many_arg(uint p0) {
  return __builtin_hlsl_asdouble(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

double test_non_matching(uint p0, uint2 p1) {
  return __builtin_hlsl_asdouble(p0, p1);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_asdouble' must have the same type}}
}

double test_non_uint(uint64_t p0) {
  return __builtin_hlsl_asdouble(p0, p0);
  // expected-error@-1 {{invalid operand of type 'uint64_t' (aka 'unsigned long') where 'unsigned int' or a vector of such type is required}}
}
