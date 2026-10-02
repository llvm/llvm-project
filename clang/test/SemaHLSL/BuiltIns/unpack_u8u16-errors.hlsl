// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

void test_builtin_no_args() {
  __builtin_hlsl_unpack_u8u16();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

uint16_t4 test_builtin_extra_args(uint8_t4_packed p0) {
  return __builtin_hlsl_unpack_u8u16(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

uint16_t4 test_builtin_int_arg(int p0) {
  return __builtin_hlsl_unpack_u8u16(p0);
  // expected-error@-1 {{passing 'int' to parameter of incompatible type 'uint8_t4_packed'}}
}

uint16_t4 test_builtin_float_arg(float p0) {
  return __builtin_hlsl_unpack_u8u16(p0);
  // expected-error@-1 {{passing 'float' to parameter of incompatible type 'uint8_t4_packed'}}
}
