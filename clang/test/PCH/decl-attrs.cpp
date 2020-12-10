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

#else

namespace preferred_name {
  Z z;

  template<typename T> T forget(T t) { return t; }
  void f() {
    forget(y).foo(); // expected-error {{no member named 'foo' in 'preferred_name::Y'}}
    forget(z).foo(); // expected-error {{no member named 'foo' in 'preferred_name::Z'}}
  }
}

#endif
