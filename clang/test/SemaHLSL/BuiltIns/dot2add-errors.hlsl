// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

float test_too_few_arg() {
  return dot2add();
  // expected-error@-1 {{no matching function for call to 'dot2add'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 0 were provided}}
}

float test_too_many_arg(half2 p1, half2 p2, float p3) {
  return dot2add(p1, p2, p3, p1);
  // expected-error@-1 {{no matching function for call to 'dot2add'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function not viable: requires 3 arguments, but 4 were provided}}
}
