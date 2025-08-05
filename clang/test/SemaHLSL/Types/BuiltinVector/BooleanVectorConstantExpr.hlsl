// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

// expected-no-diagnostics

export void fn() {
  _Static_assert((true.xxxx).y == true, "Woo!");

  _Static_assert((true.xx).x && false == false, "Woo!");
}
