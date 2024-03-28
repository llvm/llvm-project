// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -verify


double2 test_int_builtin(double2 p0) {
    return __builtin_elementwise_bitreverse(p0);
  // expected-error@-1 {{1st argument must be a vector of integers (was 'double2' (aka 'vector<double, 2>'))}}
}

int2 test_int_builtin(int2 p0) {
    return __builtin_elementwise_bitreverse(p0);
  // expected-error@-1 {{passing 'int2' (aka 'vector<int, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int' (vector of 2 'unsigned int' values)}}
}
