// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

export void fn1() {
  int2 A = {1,2};
  int X = A[-1];
  // expected-error@-1 {{vector element index -1 is out of bounds}}
}

export void fn2() {
  int4 A = {1,2,3,4};
  int X = A[4];
  // expected-error@-1 {{vector element index 4 is out of bounds}}
}

export void fn3() {
  bool2 A = {true,true};
  bool X = A[-1];
  // expected-error@-1 {{vector element index -1 is out of bounds}}
}
