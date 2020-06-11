// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { 
  void f(A* p = this) { }	// expected-error{{invalid use of 'this'}}

  void test();
};

void A::test() {
  void g(int = this); // expected-error {{default argument references 'this'}}
}
