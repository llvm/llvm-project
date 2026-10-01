// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

void test_builtin_no_args() {
  __builtin_hlsl_pack_clamp_u8();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint8_t4_packed test_builtin_extra_args(int32_t4 p0) {
  return __builtin_hlsl_pack_clamp_u8(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

uint8_t4_packed test_builtin_64bit_arg(int64_t4 p0) {
  return __builtin_hlsl_pack_clamp_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of signed integer types (was 'int64_t4' (aka 'vector<int64_t, 4>'))}}
}

uint8_t4_packed test_builtin_float_vec_arg(float32_t4 p0) {
  return __builtin_hlsl_pack_clamp_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of signed integer types (was 'float32_t4' (aka 'vector<float32_t, 4>'))}}
}

uint8_t4_packed test_builtin_wrong_vec_elems(int32_t3 p0) {
  return __builtin_hlsl_pack_clamp_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of signed integer types (was 'int32_t3' (aka 'vector<int32_t, 3>'))}}
}

uint8_t4_packed test_builtin_scalar_arg(int p0) {
  return __builtin_hlsl_pack_clamp_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of signed integer types (was 'int')}}
}
