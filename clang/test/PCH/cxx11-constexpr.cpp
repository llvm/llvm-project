// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t -verify %s

// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch -fpch-instantiate-templates %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED

struct B {
  B();
  constexpr B(char) {}
};

struct C {
  B b;
  double d = 0.0;
};

struct D : B {
  constexpr D(int n) : B('x'), k(2*n+1) {}
  int k;
};

constexpr int value = 7;

template<typename T>
constexpr T plus_seven(T other) {
  return value + other;
}

#else

static_assert(D(4).k == 9, "");
constexpr int f(C c) { return 0; } // expected-error {{not a literal type}}
// expected-note@16 {{not an aggregate and has no constexpr constructors}}
constexpr B b; // expected-error {{constant expression}} expected-note {{non-constexpr}}
               // expected-note@12 {{here}}

static_assert(plus_seven(3) == 10, "");

#endif
