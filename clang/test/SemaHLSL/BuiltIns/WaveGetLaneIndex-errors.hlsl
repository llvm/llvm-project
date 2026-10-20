// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

int test_too_many_arg(int x) {
  return __builtin_hlsl_wave_get_lane_index(x);
  // expected-error@-1 {{too many arguments to function call, expected 0, have 1}}
}
