// RUN: %clang_cc1 -finclude-default-header
// -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only
// -disable-llvm-passes -verify

double2 test_int_builtin(double2 p0) {
  return __builtin_hlsl_elementwise_countbits(p0);
  // expected-error@-1 {{passing 'double2' (aka 'vector<double, 2>') to
  // parameter of incompatible type
  // '__attribute__((__vector_size__(2 * sizeof(int)))) int'
  // (vector of 2 'int' values)}}
}

float test_ambiguous(float p0) {
  return countbits(p0);
  // expected-error@-1 {{call to 'countbits' is ambiguous}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}  
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}  
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function}}
}

float test_float_builtin(float p0) {
  return __builtin_hlsl_elementwise_countbits(p0);
  // expected-error@-1 {{passing 'double' to parameter of incompatible type
  // 'int'}}
}
