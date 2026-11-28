// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library -fsyntax-only -verify %s

float test_no_second_arg(float X) {
  return frexp(X);
  // expected-error@-1 {{no matching function for call to 'frexp'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* 4 {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* 2 {{candidate function template not viable: requires 2 arguments, but 1 was provided}}
}

float test_too_many_args(float X) {
  float E, F;
  return frexp(X, E, F);
  // expected-error@-1 {{no matching function for call to 'frexp'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* 4 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@hlsl/hlsl_intrinsics.h:* 2 {{candidate function template not viable: requires 2 arguments, but 3 were provided}}
}

float test_exp_rvalue(float X) {
  return frexp(X, 1.0f);
  // expected-error@-1 {{cannot bind non-lvalue argument '1.F' to out parameter}}
}

float test_exp_const(float X) {
  const float E = 0;
  return frexp(X, E);
  // expected-error@-1 {{read-only variable is not assignable}}
}

float3 test_vector_exp_rvalue(float3 X) {
  return frexp(X, X * 2);
  // expected-error@-1 {{cannot bind non-lvalue argument 'X * 2' to out parameter}}
}

vector<float, 8> test_long_vector_exp_rvalue(vector<float, 8> X) {
  return frexp(X, X * 2);
  // expected-error@-1 {{cannot bind non-lvalue argument 'X * 2' to out parameter}}
}

// Only float is supported, so other types are converted, as with DXC.
float test_double_input(double X) {
  float E;
  return frexp(X, E);
  // expected-warning@-1 {{implicit conversion loses floating-point precision: 'double' to 'float'}}
}

float test_int_input(int X) {
  float E;
  return frexp(X, E);
  // expected-warning@-1 {{implicit conversion from 'int' to 'float' may lose precision}}
}
