// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -fsyntax-only -verify
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library %s -fnative-half-type -fsyntax-only -verify

// expected-no-diagnostics

void test_long_vector_types(vector<half, 5> H, vector<float, 5> F) {
  static_assert(__is_same(decltype(normalize(H)), vector<half, 5>), "");
  static_assert(__is_same(decltype(normalize(F)), vector<float, 5>), "");
  static_assert(__is_same(decltype(length(H)), half), "");
  static_assert(__is_same(decltype(length(F)), float), "");
  static_assert(__is_same(decltype(distance(H, H)), half), "");
  static_assert(__is_same(decltype(distance(F, F)), float), "");
}
