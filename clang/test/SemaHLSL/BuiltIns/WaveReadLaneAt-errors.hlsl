// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

bool test_too_few_arg() {
  return __builtin_hlsl_wave_read_lane_at();
  // expected-error@-1 {{too few arguments to function call, expected 2, have 0}}
}

float2 test_too_few_arg_1(float2 p0) {
  return __builtin_hlsl_wave_read_lane_at(p0);
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
}

float2 test_too_many_arg(float2 p0) {
  return __builtin_hlsl_wave_read_lane_at(p0, p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

float3 test_index_type_check(float3 p0, double idx) {
  return __builtin_hlsl_wave_read_lane_at(p0, idx);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'unsigned int'}}
}
