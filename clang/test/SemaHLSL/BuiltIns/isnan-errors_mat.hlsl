// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify

bool2x2 test_builtin_isnan_double_matrix(double2x2 p0) {
  return __builtin_hlsl_elementwise_isnan(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double2x2' (aka 'matrix<double, 2, 2>'))}}
}
