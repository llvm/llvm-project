// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

// The constructors are invalid, so the classes are still aggregates.

struct A {
  A() && : A{} {} // expected-error{{ref-qualifier '&&' is not allowed on a constructor}}
};

struct B {
  int x;
  B() const : B{1} {} // expected-error{{'const' qualifier is not allowed on a constructor}}
};

struct S {
  static S() : S{} {} // expected-error{{constructor cannot be declared 'static'}}
};

struct M {
  int x;
  M() const : M{1} {} // expected-error{{'const' qualifier is not allowed on a constructor}}
  M(int) volatile : M{} {} // expected-error{{'volatile' qualifier is not allowed on a constructor}}
  static M(int, int) : M{} {} // expected-error{{constructor cannot be declared 'static'}}
};

#if __cplusplus >= 202002L
struct C {
  int x;
  C() & : C(1) {} // expected-error{{ref-qualifier '&' is not allowed on a constructor}}
};
#endif

#if __cplusplus >= 202302L
struct E {
  E(this E &) : E{} {} // expected-error{{an explicit object parameter cannot appear in a constructor}}
};
#endif
