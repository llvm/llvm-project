// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

int2x2 test_too_many_arg(float2x2 p0) {
  return __builtin_hlsl_elementwise_sign(p0, p0);
  // expected-error@-1 2 {{cannot pass object of non-trivial type 'float2x2' (aka 'matrix<float, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@-2 {{too many arguments to function call, expected 1, have 2}}
}

int2x2 test_bool_matrix(bool2x2 p0) {
  return __builtin_hlsl_elementwise_sign(p0);
  // expected-error@-1 {{cannot pass object of non-trivial type 'bool2x2' (aka 'matrix<bool, 2, 2>') through variadic function; call will abort at runtime}}
  // expected-error@-2 {{1st argument must be a scalar or vector of integer or floating-point types (was 'bool2x2' (aka 'matrix<bool, 2, 2>'))}}
}
