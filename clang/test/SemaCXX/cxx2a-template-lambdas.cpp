// RUN: %clang_cc1 -std=c++2a -verify %s

template<typename, typename>
constexpr bool is_same = false;

template<typename T>
constexpr bool is_same<T, T> = true;

template<typename T>
struct DummyTemplate { };

void func() {
  auto L0 = []<typename T>(T arg) {
    static_assert(is_same<T, int>); // expected-error {{static assertion failed}}
  };
  L0(0);
  L0(0.0); // expected-note {{in instantiation}}

  auto L1 = []<int I> {
    static_assert(I == 5); // expected-error {{static assertion failed}}
  };
  L1.operator()<5>();
  L1.operator()<6>(); // expected-note {{in instantiation}}

  auto L2 = []<template<typename> class T, class U>(T<U> &&arg) {
    static_assert(is_same<T<U>, DummyTemplate<float>>); // // expected-error {{static assertion failed}}
  };
  L2(DummyTemplate<float>());
  L2(DummyTemplate<double>()); // expected-note {{in instantiation}}
}

template<typename T> // expected-note {{declared here}}
struct ShadowMe {
  void member_func() {
    auto L = []<typename T> { }; // expected-error {{'T' shadows template parameter}}
  }
};

template<typename T>
constexpr T outer() {
  return []<T x>() { return x; }.template operator()<123>(); // expected-error {{no matching member function}} \
                                                                expected-note {{candidate template ignored}}
}
static_assert(outer<int>() == 123);
template int *outer<int *>(); // expected-note {{in instantiation}}
