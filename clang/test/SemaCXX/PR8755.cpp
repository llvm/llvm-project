// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename T>
struct A {
  typedef int iterator;  // expected-note{{declared here}}
};

template <typename T>
void f() {
  class A <T> ::iterator foo;  // expected-error{{typedef 'iterator' cannot be referenced with a class specifier}}
}

void g() {
  f<int>();  // expected-note{{in instantiation of function template}}
}

