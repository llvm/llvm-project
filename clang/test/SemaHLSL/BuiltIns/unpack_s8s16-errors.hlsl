// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

void test_builtin_no_args() {
  __builtin_hlsl_unpack_s8s16();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

int16_t4 test_builtin_extra_args(int8_t4_packed p0) {
  return __builtin_hlsl_unpack_s8s16(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

int16_t4 test_builtin_int_arg(int p0) {
  return __builtin_hlsl_unpack_s8s16(p0);
  // expected-error@-1 {{passing 'int' to parameter of incompatible type 'int8_t4_packed'}}
}

int16_t4 test_builtin_float_arg(float p0) {
  return __builtin_hlsl_unpack_s8s16(p0);
  // expected-error@-1 {{passing 'float' to parameter of incompatible type 'int8_t4_packed'}}
}
