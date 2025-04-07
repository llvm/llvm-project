// RUN: %clang_cc1 -std=hlsl202y -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

double4 test_double_inputs(double p0, double p1, double p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = double] has been explicitly deleted}}
}

int4 test_int_inputs(int p0, int p1, int p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = int] has been explicitly deleted}}
}

uint4 test_uint_inputs(uint p0, uint p1, uint p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = unsigned int] has been explicitly deleted}}
}

int64_t4 test_int64_t_inputs(int64_t p0, int64_t p1, int64_t p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = long] has been explicitly deleted}}
}

uint64_t4 test_uint64_t_inputs(uint64_t p0, uint64_t p1, uint64_t p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = unsigned long] has been explicitly deleted}}
}

bool4 test_bool_inputs(bool p0, bool p1, bool p2) {
  return lit(p0, p1, p2);
  // expected-error@-1 {{call to deleted function 'lit'}}
  // expected-note@hlsl/hlsl_intrinsics.h:* {{candidate function [with T = bool] has been explicitly deleted}}
}
