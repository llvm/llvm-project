// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int test_too_few_arg0() {
  return __builtin_hlsl_dot4add_i8packed();
  // expected-error@-1 {{too few arguments to function call, expected 3, have 0}}
}

int test_too_few_arg1(int p0) {
  return __builtin_hlsl_dot4add_i8packed(p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 1}}
}

int test_too_few_arg2(int p0) {
  return __builtin_hlsl_dot4add_i8packed(p0, p0);
  // expected-error@-1 {{too few arguments to function call, expected 3, have 2}}
}

int test_too_many_arg(int p0) {
  return __builtin_hlsl_dot4add_i8packed(p0, p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 3, have 4}}
}

struct S { float f; };

int test_expr_struct_type_check(S p0, int p1) {
  return __builtin_hlsl_dot4add_i8packed(p0, p1, p1);
  // expected-error@-1 {{no viable conversion from 'S' to 'unsigned int'}}
}
