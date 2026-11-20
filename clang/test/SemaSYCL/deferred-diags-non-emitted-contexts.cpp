// RUN: %clang_cc1 -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -std=c++20 -fsycl-is-device -fsyntax-only -verify %s

// Code that is never emitted for the device is not diagnosed.

int vfn(int, ...);

constexpr __float128 q = 1; // expected-note {{'q' defined here}}
struct Z { int a[0]; };
constexpr Z z{};

[[clang::sycl_external]] int constexpr_init() {
  constexpr int i = (int)q;
  return i;
}
constinit int ci = (int)q;
[[clang::sycl_external]] int constinit_init() { return ci; }

template <int N> struct S { static constexpr int value = N; };
[[clang::sycl_external]] int template_arg() { return S<(int)q>::value; }
[[clang::sycl_external]] int array_bound() {
  int a[(int)q];
  return sizeof(a);
}

[[clang::sycl_external]] int unevaluated() {
  static_assert(sizeof(z.a) == 0);
  return sizeof(vfn(1, 2)) + noexcept(vfn(1, 2));
}

template <bool B> void discarded_tmpl() {
  if constexpr (B)
    vfn(1, 2);
}
[[clang::sycl_external]] void discarded() {
  discarded_tmpl<false>();
  if constexpr (false)
    vfn(1, 2);
}

[[clang::sycl_external]] int emitted() {
  vfn(1, 2); // expected-error {{SYCL device code does not support variadic functions}}
  return (int)q; // expected-error {{'q' requires 128 bit size 'const __float128' type support, but target 'spir64-unknown-unknown' does not support it}}
}
