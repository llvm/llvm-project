// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected


uint4 test_asuint_too_many_arg(float p0, float p1) {
  return __builtin_hlsl_bit_cast_32(p0, p1);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}

uint test_asuint_double(double p1) {
    return __builtin_hlsl_bit_cast_32(p1);
    // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}


uint test_asuint_half(half p1) {
    return __builtin_hlsl_bit_cast_32(p1);
    // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}
