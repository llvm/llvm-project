// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -DTEST_FUNC=__builtin_elementwise_atan2
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -DTEST_FUNC=__builtin_elementwise_fmod
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -DTEST_FUNC=__builtin_elementwise_pow

double test_double_builtin(double p0, double p1) {
    return TEST_FUNC(p0, p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double')}}
}

double2 test_vec_double_builtin(double2 p0, double2 p1) {
    return TEST_FUNC(p0, p1);
  // expected-error@-1 {{1st argument must be a scalar or vector of 16 or 32 bit floating-point types (was 'double2' (aka 'vector<double, 2>'))}}
}
