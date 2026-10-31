// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -verify
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

float test_no_second_arg(float X) {
  return __builtin_hlsl_elementwise_frexp(X);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float test_too_many_arg(float X) {
  float E, F;
  return __builtin_hlsl_elementwise_frexp(X, E, F);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float test_first_arg_double(double X) {
  float E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'double' where 'float' or a vector or matrix of such type is required}}
}

float test_first_arg_int(int X) {
  float E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'int' where 'float' or a vector or matrix of such type is required}}
}

float test_first_arg_bool(bool X) {
  float E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'bool' where 'float' or a vector or matrix of such type is required}}
}

float test_second_arg_int(float X) {
  int E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'int' where 'float' or a vector or matrix of such type is required}}
}

float2 test_shape_mismatch(float2 X) {
  float3 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_elementwise_frexp' must have the same type}}
}

float test_second_arg_rvalue(float X) {
  return __builtin_hlsl_elementwise_frexp(X, 1.0f);
  // expected-error@-1 {{cannot bind non-lvalue argument '1.F' to out parameter}}
}

float test_second_arg_const(float X) {
  const float E = 0;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{cannot bind non-lvalue argument 'E' to out parameter}}
}

float test_overload_rvalue(float X) {
  return frexp(X, 1.0f);
  // expected-error@-1 {{cannot bind non-lvalue argument '1.F' to out parameter}}
}

half test_builtin_half(half X) {
  half E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'half' where 'float' or a vector or matrix of such type is required}}
}
