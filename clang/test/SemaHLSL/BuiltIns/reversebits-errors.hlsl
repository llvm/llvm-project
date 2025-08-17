// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify


double2 test_int_builtin(double2 p0) {
    return __builtin_elementwise_bitreverse(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of integer types (was 'double2' (aka 'vector<double, 2>'))}}
}

int2 test_int_builtin(int2 p0) {
    return __builtin_elementwise_bitreverse(p0);
  // expected-error@-1 {{1st argument must be a scalar or vector of unsigned integer types (was 'int2' (aka 'vector<int, 2>'))}}
}
