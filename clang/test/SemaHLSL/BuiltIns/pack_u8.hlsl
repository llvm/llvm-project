// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

void test_no_args() {
  pack_u8();
  // expected-error@-1 {{no matching function for call to 'pack_u8'}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 2 {{candidate function not viable: requires 1 argument, but 0 were provided}}
}

uint8_t4_packed test_extra_args(uint32_t4 p0) {
  return pack_u8(p0, p0);
  // expected-error@-1 {{no matching function for call to 'pack_u8'}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 2 {{candidate function not viable: requires 1 argument, but 2 were provided}}
}

uint8_t4_packed test_64bit_arg(uint64_t4 p0) {
  return pack_u8(p0);
  // expected-error@-1 {{call to 'pack_u8' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 2 {{candidate function}}
}

uint8_t4_packed test_float_vec_arg(float32_t4 p0) {
  return pack_u8(p0);
  // expected-error@-1 {{call to 'pack_u8' is ambiguous}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 2 {{candidate function}}
}

uint8_t4_packed test_wrong_vec_elems(uint32_t3 p0) {
  return pack_u8(p0);
  // expected-error@-1 {{no matching function for call to 'pack_u8'}}
  // expected-note@hlsl/hlsl_alias_intrinsics_gen.inc:* 2 {{candidate function not viable: no known conversion from}}
}

void test_builtin_no_args() {
  __builtin_hlsl_pack_u8();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint8_t4_packed test_builtin_extra_args(uint32_t4 p0) {
  return __builtin_hlsl_pack_u8(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

uint8_t4_packed test_builtin_64bit_arg(uint64_t4 p0) {
  return __builtin_hlsl_pack_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of unsigned integer types (was 'uint64_t4' (aka 'vector<uint64_t, 4>'))}}
}

uint8_t4_packed test_builtin_float_vec_arg(float32_t4 p0) {
  return __builtin_hlsl_pack_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of unsigned integer types (was 'float32_t4' (aka 'vector<float32_t, 4>'))}}
}

uint8_t4_packed test_builtin_wrong_vec_elems(uint32_t3 p0) {
  return __builtin_hlsl_pack_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of unsigned integer types (was 'uint32_t3' (aka 'vector<uint32_t, 3>'))}}
}

uint8_t4_packed test_builtin_scalar_arg(uint p0) {
  return __builtin_hlsl_pack_u8(p0);
  // expected-error@-1 {{1st argument must be a vector of unsigned integer types (was 'uint' (aka 'unsigned int'))}}
}
