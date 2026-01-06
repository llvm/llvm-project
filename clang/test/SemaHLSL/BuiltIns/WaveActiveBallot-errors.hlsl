// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

uint4 test_too_few_arg() {
  return __builtin_hlsl_wave_active_ballot();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint4 test_too_many_arg(bool p0) {
  return __builtin_hlsl_wave_active_ballot(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct Foo
{
  int a;
};

uint4 test_type_check(Foo p0) {
  return __builtin_hlsl_wave_active_ballot(p0);
  // expected-error@-1 {{no viable conversion from 'Foo' to 'bool'}}
}
