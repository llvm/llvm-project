// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> struct A {
  enum E : T;
  E v = E{};
};

template<typename T> enum A<int>::E : T { e1 }; // expected-error {{template parameter list matching the non-templated nested type 'A<int>' should be empty ('template<>')}}

A<int> a;
