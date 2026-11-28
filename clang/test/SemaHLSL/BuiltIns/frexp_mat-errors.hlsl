// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library -fsyntax-only -verify %s

float2x2 test_exp_rvalue(float2x2 X) {
  return frexp(X, X * 2);
  // expected-error@-1 {{cannot bind non-lvalue argument 'X * 2' to out parameter}}
}

float2x2 test_exp_const(float2x2 X) {
  const float2x2 E = 0;
  return frexp(X, E);
  // expected-error@-1 {{read-only variable is not assignable}}
}
