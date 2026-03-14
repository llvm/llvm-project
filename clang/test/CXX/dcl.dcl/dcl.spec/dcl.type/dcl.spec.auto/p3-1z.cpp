// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1z

// FIXME: This is in p10 (?) in C++1z.
template<typename T> struct A {
  A(T);
};
template<typename T> A(T) -> A<T>;
A a = a; // expected-error{{variable 'a' declared with deduced type 'A' cannot appear in its own initializer}}
