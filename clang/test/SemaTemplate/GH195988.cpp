// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename> struct A {
  template <typename T> static B x; // expected-error {{unknown type name 'B'}}
  template <typename T> static int x<T*>;
};

A<int> a;
