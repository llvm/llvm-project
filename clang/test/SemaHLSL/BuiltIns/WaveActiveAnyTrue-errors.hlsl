// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

bool test_too_few_arg() {
  return __builtin_hlsl_wave_active_any_true();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

bool test_too_many_arg(bool p0) {
  return __builtin_hlsl_wave_active_any_true(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct Foo
{
  int a;
};

bool test_type_check(Foo p0) {
  return __builtin_hlsl_wave_active_any_true(p0);
  // expected-error@-1 {{no viable conversion from 'Foo' to 'bool'}}
}
