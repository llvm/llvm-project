// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

float2x2 test_builtin_int_matrix(int2x2 X) {
  float2x2 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'int2x2' (aka 'matrix<int, 2, 2>') where 'float' or a vector or matrix of such type is required}}
}

float2x2 test_builtin_bool_matrix(bool2x2 X) {
  float2x2 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'bool2x2' (aka 'matrix<bool, 2, 2>') where 'float' or a vector or matrix of such type is required}}
}

float2x2 test_builtin_double_matrix(double2x2 X) {
  float2x2 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'double2x2' (aka 'matrix<double, 2, 2>') where 'float' or a vector or matrix of such type is required}}
}

half2x2 test_builtin_half_matrix(half2x2 X) {
  half2x2 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'half2x2' (aka 'matrix<half, 2, 2>') where 'float' or a vector or matrix of such type is required}}
}

float2x2 test_builtin_exp_int_matrix(float2x2 X) {
  int2x2 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{invalid operand of type 'int2x2' (aka 'matrix<int, 2, 2>') where 'float' or a vector or matrix of such type is required}}
}

float2x2 test_builtin_dim_mismatch(float2x2 X) {
  float2x3 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_elementwise_frexp' must have the same type}}
}

float2x2 test_builtin_matrix_vector_mismatch(float2x2 X) {
  float4 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_elementwise_frexp' must have the same type}}
}

float2x2 test_builtin_exp_not_lvalue(float2x2 X) {
  return __builtin_hlsl_elementwise_frexp(X, X * 2);
  // expected-error@-1 {{cannot bind non-lvalue argument 'X * 2' to out parameter}}
}

float2x2 test_overload_exp_not_lvalue(float2x2 X) {
  return frexp(X, X * 2);
  // expected-error@-1 {{cannot bind non-lvalue argument 'X * 2' to out parameter}}
}
