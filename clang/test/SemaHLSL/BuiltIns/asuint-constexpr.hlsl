// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -verify

// expected-no-diagnostics

// Because asuint should be constant evaluated, all the static asserts below
// should work!
void ConstExprTest() {
  static_assert(asuint(1) == 1u, "One");
  static_assert(asuint(2.xxx).y == 2u, "Two");
}
