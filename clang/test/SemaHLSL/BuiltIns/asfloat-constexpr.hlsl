// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

// expected-no-diagnostics

// Because asfloat should be constant evaluated, all the static asserts below
// should work!
void ConstExprTest() {
  static_assert(asfloat(0x3f800000) == 1.0f, "One");
  static_assert(asfloat(0x40000000.xxx).y == 2.0f, "Two");
}
