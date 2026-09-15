// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// The constructors are invalid, so the classes are still aggregates.

struct A {
  A() && : A{} {} // expected-error{{ref-qualifier '&&' is not allowed on a constructor}}
};

struct B {
  int x;
  B() const : B{1} {} // expected-error{{'const' qualifier is not allowed on a constructor}}
};

#if __cplusplus >= 202002L
struct C {
  int x;
  C() & : C(1) {} // expected-error{{ref-qualifier '&' is not allowed on a constructor}}
};
#endif
