// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -O1 -verify

bool test_too_few_arg(bool a) {
  return __builtin_hlsl_and(a);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

bool test_too_many_arg(bool a) {
  return __builtin_hlsl_and(a, a, a);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

bool2 test_mismatched_args(bool2 a, bool3 b) {
  return __builtin_hlsl_and(a, b);
  // expected-error@-1 {{all arguments to '__builtin_hlsl_and' must have the same type}}
}

bool test_incorrect_type(int a) {
  return __builtin_hlsl_and(a, a);
  // expected-error@-1{{invalid operand of type 'int' where 'bool' or a vector of such type is required}}
}
