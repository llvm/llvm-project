// RUN: %clang_cc1 -std=c++1z -verify %s

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename T> struct initializer_list {
    const T *p;
    size_t n;
    initializer_list();
  };
  // FIXME: This should probably not be necessary.
  template<typename T> initializer_list(initializer_list<T>) -> initializer_list<T>;
}

template<typename T> constexpr bool has_type(...) { return false; }
template<typename T> constexpr bool has_type(T) { return true; }

std::initializer_list il = {1, 2, 3, 4, 5};

template<typename T> struct vector {
  template<typename Iter> vector(Iter, Iter);
  vector(std::initializer_list<T>);
};

template<typename T> vector(std::initializer_list<T>) -> vector<T>;
template<typename Iter> explicit vector(Iter, Iter) -> vector<typename Iter::value_type>;
template<typename T> explicit vector(std::size_t, T) -> vector<T>;

vector v1 = {1, 2, 3, 4};
static_assert(has_type<vector<int>>(v1));

struct iter { typedef char value_type; } it, end;
vector v2(it, end);
static_assert(has_type<vector<char>>(v2));

vector v3(5, 5);
static_assert(has_type<vector<int>>(v3));

vector v4 = {it, end};
static_assert(has_type<vector<iter>>(v4));

vector v5{it, end};
static_assert(has_type<vector<iter>>(v5));

template<typename ...T> struct tuple { tuple(T...); };
template<typename ...T> explicit tuple(T ...t) -> tuple<T...>; // expected-note {{declared}}
// FIXME: Remove
template<typename ...T> tuple(tuple<T...>) -> tuple<T...>;

const int n = 4;
tuple ta = tuple{1, 'a', "foo", n};
static_assert(has_type<tuple<int, char, const char*, int>>(ta));

tuple tb{ta};
static_assert(has_type<tuple<int, char, const char*, int>>(tb));

// FIXME: This should be tuple<tuple<...>>; when the above guide is removed.
tuple tc = {ta};
static_assert(has_type<tuple<int, char, const char*, int>>(tc));

tuple td = {1, 2, 3}; // expected-error {{selected an explicit deduction guide}}
static_assert(has_type<tuple<int, char, const char*, int>>(td));

// FIXME: This is a GCC extension for now; if CWG don't allow this, at least
// add a warning for it.
namespace new_expr {
  tuple<int> *p = new tuple{0};
  tuple<float, float> *q = new tuple(1.0f, 2.0f);
}

namespace ambiguity {
  template<typename T> struct A {};
  A(unsigned short) -> A<int>; // expected-note {{candidate}}
  A(short) -> A<int>; // expected-note {{candidate}}
  A a = 0; // expected-error {{ambiguous deduction for template arguments of 'A'}}

  template<typename T> struct B {};
  template<typename T> B(T(&)(int)) -> B<int>; // expected-note {{candidate function [with T = int]}}
  template<typename T> B(int(&)(T)) -> B<int>; // expected-note {{candidate function [with T = int]}}
  int f(int);
  B b = f; // expected-error {{ambiguous deduction for template arguments of 'B'}}
}

// FIXME: Revisit this once CWG decides if attributes, and [[deprecated]] in
// particular, should be permitted here.
namespace deprecated {
  template<typename T> struct A { A(int); };
  [[deprecated]] A(int) -> A<void>; // expected-note {{marked deprecated here}}
  A a = 0; // expected-warning {{'<deduction guide for A>' is deprecated}}
}

namespace dependent {
  template<template<typename...> typename A> decltype(auto) a = A{1, 2, 3};
  static_assert(has_type<vector<int>>(a<vector>));
  static_assert(has_type<tuple<int, int, int>>(a<tuple>));

  struct B {
    template<typename T> struct X { X(T); };
    X(int) -> X<int>;
    template<typename T> using Y = X<T>; // expected-note {{template}}
  };
  template<typename T> void f() {
    typename T::X tx = 0;
    typename T::Y ty = 0; // expected-error {{alias template 'Y' requires template arguments; argument deduction only allowed for class templates}}
  }
  template void f<B>(); // expected-note {{in instantiation of}}

  template<typename T> struct C { C(T); };
  template<typename T> C(T) -> C<T>;
  template<typename T> void g(T a) {
    C b = 0;
    C c = a;
    using U = decltype(b); // expected-note {{previous}}
    using U = decltype(c); // expected-error {{different types ('C<const char *>' vs 'C<int>')}}
  }
  void h() {
    g(0);
    g("foo"); // expected-note {{instantiation of}}
  }
}

namespace look_into_current_instantiation {
  template<typename U> struct Q {};
  template<typename T> struct A {
    using U = T;
    template<typename> using V = Q<A<T>::U>;
    template<typename W = int> A(V<W>);
  };
  A a = Q<float>(); // ok, can look through class-scope typedefs and alias
                    // templates, and members of the current instantiation
  A<float> &r = a;

  template<typename T> struct B { // expected-note {{could not match 'B<T>' against 'int'}}
    struct X {
      typedef T type;
    };
    B(typename X::type); // expected-note {{couldn't infer template argument 'T'}}
  };
  B b = 0; // expected-error {{no viable}}

  // We should have a substitution failure in the immediate context of
  // deduction when using the C(T, U) constructor (probably; core wording
  // unclear).
  template<typename T> struct C {
    using U = typename T::type;
    C(T, U);
  };

  struct R { R(int); typedef R type; };
  C(...) -> C<R>;

  C c = {1, 2};
}
