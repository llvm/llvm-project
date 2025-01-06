
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected

bool test_too_few_arg() {
  return __builtin_hlsl_any();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
}

bool test_too_many_arg(float2 p0) {
  return __builtin_hlsl_any(p0, p0);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
}
