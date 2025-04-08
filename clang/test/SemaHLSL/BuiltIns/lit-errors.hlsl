// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected=note

float4 test_double_inputs(double p0, double p1, double p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to 'lit' is ambiguous}}
}

float4 test_int_inputs(int p0, int p1, int p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to 'lit' is ambiguous}}
}

float4 test_bool_inputs(bool p0, bool p1, bool p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to 'lit' is ambiguous}}
}
