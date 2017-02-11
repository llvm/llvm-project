// RUN: %clang_cc1 -std=c++1z -verify %s

namespace std_example {
  template <class T> struct A { // expected-note 2{{candidate}}
    // FIXME: This is a bad way to diagnose redeclaration of a class member!
    explicit A(const T &, ...) noexcept; // expected-note {{previous}} expected-note {{candidate}}
    A(T &&, ...); // expected-error {{missing exception specification 'noexcept'}}
  };

  int i;
  // FIXME: All but the first should be valid once we synthesize deduction guides from constructors.
  A a1 = {i, i}; // expected-error {{no viable constructor or deduction guide}}
  A a2{i, i}; // expected-error {{no viable constructor or deduction guide}}
  A a3{0, i}; // expected-error {{no viable constructor or deduction guide}}
  A a4 = {0, i}; // expected-error {{no viable constructor or deduction guide}}

  template <class T> A(const T &, const T &) -> A<T &>;
  template <class T> explicit A(T &&, T &&) -> A<T>; // expected-note {{explicit deduction guide declared here}}

  A a5 = {0, 1}; // expected-error {{class template argument deduction for 'A' selected an explicit deduction guide}}
  A a6{0, 1};
  A a7 = {0, i}; // expected-note {{in instantiation of}}
  A a8{0, i}; // expected-error {{no matching constructor}}

  template <class T> struct B {
    template <class U> using TA = T;
    template <class U> B(U, TA<U>);
  };
  // FIXME: This is valid.
  B b{(int *)0, (char *)0}; // expected-error {{no viable constructor or deduction guide}}
}

namespace check {
  using namespace std_example;
  template<typename T, typename U> constexpr bool same = false;
  template<typename T> constexpr bool same<T, T> = true;

  static_assert(same<decltype(a2), A<int>>);
  static_assert(same<decltype(a3), A<int>>);
  static_assert(same<decltype(a4), A<int>>);
  static_assert(same<decltype(a6), A<int>>);
  static_assert(same<decltype(b), A<char*>>);
}
