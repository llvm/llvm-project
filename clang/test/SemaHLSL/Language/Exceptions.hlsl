// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -fcxx-exceptions -fsyntax-only -verify %s

void test_try() {
  try { // expected-error {{cannot use 'try' in HLSL}}
  } catch (...) {
  }
}

void test_throw() {
  throw 1; // expected-error {{cannot use 'throw' in HLSL}}
}

void test_rethrow() {
  throw; // expected-error {{cannot use 'throw' in HLSL}}
}
