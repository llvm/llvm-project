// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -fsyntax-only -verify

bool test_too_few_args() {
  return __builtin_hlsl_quad_all();
  // expected-error@-1 {{too few arguments to function call, expected 1}}
}

bool test_too_many_args(bool cond) {
  return __builtin_hlsl_quad_all(cond, cond);
  // expected-error@-1 {{too many arguments to function call, expected 1}}
}

struct Foo {
  int value;
};

bool test_type(Foo value) {
  return __builtin_hlsl_quad_all(value);
  // expected-error@-1 {{cannot initialize a parameter of type 'bool'}}
}

bool test_vector(bool2 value) {
  return __builtin_hlsl_quad_all(value);
  // expected-warning@-1 {{implicit conversion turns vector to scalar}}
}
