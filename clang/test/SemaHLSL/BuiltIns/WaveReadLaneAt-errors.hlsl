// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -emit-llvm-only -disable-llvm-passes -verify

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

float3 test_index_double_type_check(float3 p0, double idx) {
  return __builtin_hlsl_wave_read_lane_at(p0, idx);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type 'unsigned int'}}
}

float3 test_index_int3_type_check(float3 p0, int3 idxs) {
  return __builtin_hlsl_wave_read_lane_at(p0, idxs);
  // expected-error@-1 {{passing 'int3' (aka 'vector<int, 3>') to parameter of incompatible type 'unsigned int'}}
}

struct S { float f; };

float3 test_index_S_type_check(float3 p0, S idx) {
  return __builtin_hlsl_wave_read_lane_at(p0, idx);
  // expected-error@-1 {{passing 'S' to parameter of incompatible type 'unsigned int'}}
}

S test_expr_struct_type_check(S p0, int idx) {
  return __builtin_hlsl_wave_read_lane_at(p0, idx);
  // expected-error@-1 {{invalid operand of type 'S' where a scalar or vector is required}}
}
