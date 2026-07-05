// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t.a %s
// RUN: %clang_cc1 -std=c++20 -include-pch %t.a %s -verify

#ifndef HEADER
#define HEADER

namespace preferred_name {
  template<typename T> struct X;
  using Y = X<int>;
  using Z = X<float>;
  template<typename T> struct [[using clang: preferred_name(Y), preferred_name(Z)]] X {};
  Y y;
}

namespace aligned {
  // PR48434: ensure attributes don't introduce deserialization cycles.
  template<typename T> struct X1;
  using Y1 = X1<int>;
  template<typename T> struct alignas(Y1*) X1 {};
  Y1 y1;

  template<typename T> struct X2;
  using Y2 = X2<int>;
  template<typename T> struct alignas(Y2*) X2 {};
}

#else

namespace preferred_name {
  Z z;

  template<typename T> T forget(T t) { return t; }
  void f() {
    forget(y).foo(); // expected-error {{no member named 'foo' in 'preferred_name::Y'}}
    forget(z).foo(); // expected-error {{no member named 'foo' in 'preferred_name::Z'}}
  }
}

namespace aligned {
  extern Y1 y1;
  extern Y2 y2;
  static_assert(alignof(Y1) == alignof(Y1*), "");
  static_assert(alignof(Y2) == alignof(Y2*), "");
}

#endif
