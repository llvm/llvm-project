// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fnative-int16-type -emit-llvm-only -disable-llvm-passes -verify


double2x2 test_vec_double_builtin(double2x2 p0, double2x2 p1) {
    return __builtin_elementwise_atan2(p0, p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double2x2' (aka 'matrix<double, 2, 2>'))}}
}
