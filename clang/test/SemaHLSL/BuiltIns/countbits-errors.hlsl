// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify -verify-ignore-unexpected


double test_int_builtin(double p0) {
  return countbits(p0);
  // expected-error@-1 {{call to 'countbits' is ambiguous}}
}

double2 test_int_builtin_2(double2 p0) {
  return countbits(p0);
  // expected-error@-1 {{call to 'countbits' is ambiguous}}
}

double test_int_builtin_3(float p0) {
  return countbits(p0);
  // expected-error@-1 {{call to 'countbits' is ambiguous}}
}
