// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { 
  virtual void f(int a = 7); // expected-note{{'A::f' declared here}}
}; 

struct B : public A {
  void f(int a);
}; 

void m() {
  B* pb = new B; 
  A* pa = pb; 
  pa->f(); // OK, calls pa->B::f(7) 
  pb->f(); // expected-error{{too few arguments to function call, expected 1, have 0; did you mean 'A::f'?}}
}
