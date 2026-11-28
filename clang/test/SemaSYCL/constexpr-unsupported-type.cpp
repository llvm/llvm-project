// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -std=c++20 -fsyntax-only -verify %s

// Unsupported types are diagnosed in runtime code, not in
// constant-evaluated code.

constexpr __float128 q = 1; // expected-note {{'q' defined here}}
template <int N> struct S { static constexpr int value = N; };

[[clang::sycl_external]] int d() {
  constexpr int i = (int)q;
  int a = S<(int)q>::value;
  return i + a + (int)q; // expected-error {{'q' requires 128 bit size 'const __float128' type support, but target 'spir64-unknown-unknown' does not support it}}
}
