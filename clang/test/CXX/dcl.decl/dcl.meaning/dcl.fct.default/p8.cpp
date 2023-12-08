// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { 
  void f(A* p = this) { }	// expected-error{{invalid use of 'this'}}

  void test();
};

void A::test() {
  void g(int = this);
  // expected-error@-1 {{cannot initialize a parameter of type 'int' with an rvalue of type 'A *'}}
  // expected-note@-2 {{passing argument to parameter here}}

  void h(int = ((void)this,42));
  // expected-error@-1 {{default argument references 'this'}}
}
