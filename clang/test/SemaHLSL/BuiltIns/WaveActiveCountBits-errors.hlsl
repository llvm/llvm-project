// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int test_too_few_arg() {
  return __builtin_hlsl_wave_active_count_bits();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

int test_too_many_arg(bool x) {
  return __builtin_hlsl_wave_active_count_bits(x, x);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

struct S { float f; };

int test_bad_conversion(S x) {
  return __builtin_hlsl_wave_active_count_bits(x);
  // expected-error@-1 {{no viable conversion from 'S' to 'bool'}}
}
