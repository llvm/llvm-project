// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -verify

double2 test_double_builtin(double2 p0, double2 p1) {
    return __builtin_elementwise_pow(p0,p1);
  // expected-error@-1 {{passing 'double2' (aka 'vector<double, 2>') to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(float)))) float' (vector of 2 'float' values)}}
}
