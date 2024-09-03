// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected


export uint4 test_asuint_too_many_arg(float p0, float p1) {
  return __builtin_hlsl_elementwise_asuint(p0, p1);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}


export uint fn(double p1) {
    return asuint(p1);
    // expected-error@-1 {{passing 'double' to parameter of incompatible type 'float'}}
}

export uint fn(half p1) {
    return asuint(p1);
    // expected-error@-1 {{passing 'half' to parameter of incompatible type 'float'}}
}
