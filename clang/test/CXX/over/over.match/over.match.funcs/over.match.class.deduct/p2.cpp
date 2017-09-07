// RUN: %clang_cc1 -verify -std=c++1z %s

namespace Explicit {
  // Each notional constructor is explicit if the function or function template
  // was generated from a constructor or deduction-guide that was declared explicit.
  template<typename T> struct A {
    A(T);
    A(T*);
  };
  template<typename T> A(T) -> A<T>;
  template<typename T> explicit A(T*) -> A<T>; // expected-note {{explicit}}

  int *p;
  A a(p);
  A b = p;
  A c{p};
  A d = {p}; // expected-error {{selected an explicit deduction guide}}

  using X = A<int>;
  using Y = A<int*>;

  using X = decltype(a);
  using Y = decltype(b);
  using X = decltype(c);
}

namespace std {
  template<typename T> struct initializer_list {
    const T *ptr;
    __SIZE_TYPE__ size;
    initializer_list();
  };
}

namespace p0702r1 {
  template<typename T> struct X { // expected-note {{candidate}}
    X(std::initializer_list<T>); // expected-note {{candidate}}
  };

  X xi = {0};
  X xxi = {xi};
  extern X<int> xi;
  // Prior to P0702R1, this is X<X<int>>.
  extern X<int> xxi;

  struct Y : X<int> {};
  Y y {{0}};
  X xy {y};
  extern X<int> xy;

  struct Z : X<int>, X<float> {};
  Z z = {{0}, {0.0f}};
  // This is not X<Z> even though that would work. Instead, it's ambiguous
  // between X<int> and X<float>.
  X xz = {z}; // expected-error {{no viable constructor or deduction guide}}
}
