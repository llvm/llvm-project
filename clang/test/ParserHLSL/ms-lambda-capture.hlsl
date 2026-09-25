// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -o - \
// RUN:   -fsyntax-only %s -verify

void lambda_direct_initializer() {
  // expected-warning@#lambda {{lambdas are a clang HLSL extension}}
  // expected-warning@#lambda {{lambda without a parameter clause is a C++23 extension}}
  // expected-warning@#lambda {{static lambdas are a C++23 extension}}
  int value([] static { return 1; }()); // #lambda
}
