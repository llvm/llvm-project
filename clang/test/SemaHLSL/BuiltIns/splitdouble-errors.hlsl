// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -verify

void test_no_second_arg(double D) {
  __builtin_hlsl_elementwise_splitdouble(D);
 // expected-error@-1 {{too few arguments to function call, expected 3, have 1}} 
}

void test_no_third_arg(double D) {
  uint A;
  __builtin_hlsl_elementwise_splitdouble(D, A);
 // expected-error@-1 {{too few arguments to function call, expected 3, have 2}} 
}

void test_too_many_arg(double D) {
  uint A, B, C;
  __builtin_hlsl_elementwise_splitdouble(D, A, B, C);
 // expected-error@-1 {{too many arguments to function call, expected 3, have 4}} 
}

void test_first_arg_type_mismatch(bool3 D) {
  uint3 A, B;
  __builtin_hlsl_elementwise_splitdouble(D, A, B);
 // expected-error@-1 {{invalid operand of type 'bool3' (aka 'vector<bool, 3>') where 'double' or a vector of such type is required}} 
}

void test_second_arg_type_mismatch(double D) {
  bool A;
  uint B;
  __builtin_hlsl_elementwise_splitdouble(D, A, B);
 // expected-error@-1 {{invalid operand of type 'bool' where 'unsigned int' or a vector of such type is required}} 
}

void test_third_arg_type_mismatch(double D) {
  bool A;
  uint B;
  __builtin_hlsl_elementwise_splitdouble(D, B, A);
 // expected-error@-1 {{invalid operand of type 'bool' where 'unsigned int' or a vector of such type is required}} 
}

void test_const_second_arg(double D) {
  const uint A = 1;
  uint B;
  __builtin_hlsl_elementwise_splitdouble(D, A, B);
 // expected-error@-1 {{cannot bind non-lvalue argument}} 
}

void test_const_third_arg(double D) {
  uint A;
  const uint B = 2;
  __builtin_hlsl_elementwise_splitdouble(D, A, B);
 // expected-error@-1 {{cannot bind non-lvalue argument}} 
}
