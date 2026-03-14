// RUN: %clang_cc1 -std=c++1z -verify %s

namespace std_example {
  template <class T> struct A {
    explicit A(const T &, ...) noexcept; // expected-note {{explicit}}
    A(T &&, ...); // expected-note 2{{candidate}}
  };

  int i;
  A a1 = {i, i}; // expected-error {{class template argument deduction for 'A' selected an explicit constructor for copy-list-initialization}}
  A a2{i, i};
  A a3{0, i};
  A a4 = {0, i};

  template <class T> A(const T &, const T &) -> A<T &>; // expected-note 2{{candidate}}
  template <class T> explicit A(T &&, T &&) -> A<T>; // expected-note {{explicit deduction guide declared here}}

  // FIXME: The standard gives an incorrect explanation for why a5, a7, and a8 are ill-formed.
  A a5 = {0, 1}; // expected-error {{class template argument deduction for 'A' selected an explicit deduction guide}}
  A a6{0, 1};
  A a7 = {0, i}; // expected-error {{ambiguous deduction}}
  A a8{0, i}; // expected-error {{ambiguous deduction}}

  template <class T> struct B {
    template <class U> using TA = T;
    template <class U> B(U, TA<U>);
  };
  B b{(int *)0, (char *)0};
}

namespace check {
  using namespace std_example;
  template<typename T, typename U> constexpr bool same = false;
  template<typename T> constexpr bool same<T, T> = true;

  static_assert(same<decltype(a2), A<int>>);
  static_assert(same<decltype(a3), A<int>>);
  static_assert(same<decltype(a4), A<int>>);
  static_assert(same<decltype(a6), A<int>>);
  static_assert(same<decltype(b), B<char*>>);
}
