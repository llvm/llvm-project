// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

export void cantCast() {
  int A[3] = {1,2,3};
  int B[4] = {1,2,3,4};
  B = (int[4])A;
  // expected-error@-1 {{C-style cast from 'int *' to 'int[4]' is not allowed}}
}