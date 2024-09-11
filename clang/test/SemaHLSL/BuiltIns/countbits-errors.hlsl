// RUN: %clang_cc1 -finclude-default-header
// -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only
// -disable-llvm-passes -verify


double test_int_builtin(double p0) {
  return countbits(p0);
  // expected-error@-1 {{call to 'countbits' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

double2 test_int_builtin_2(double2 p0) {
  return __builtin_elementwise_popcount(p0);
  // expected-error@-1 {{1st argument must be a vector of integers
  // (was 'double2' (aka 'vector<double, 2>'))}}
}

double test_int_builtin_3(float p0) {
  return __builtin_elementwise_popcount(p0);
  // expected-error@-1 {{1st argument must be a vector of integers
  // (was 'float')}}
}