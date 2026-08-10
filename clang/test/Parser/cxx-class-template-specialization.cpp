// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {
  template<typename T>
  void f();
};
class A::f<int>;
// expected-error@-1 {{identifier followed by '<' indicates a class template specialization but 'f' refers to a function template}}
