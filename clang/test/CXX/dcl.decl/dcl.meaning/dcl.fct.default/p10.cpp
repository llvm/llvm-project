// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { 
  virtual void f(int a = 7);
}; 

struct B : public A {
  void f(int a); // expected-note{{'f' declared here}}
}; 

void m() {
  B* pb = new B; 
  A* pa = pb; 
  pa->f(); // OK, calls pa->B::f(7) 
  pb->f(); // expected-error{{too few arguments}}
}
