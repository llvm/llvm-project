// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.cplusplus.VirtualCall -analyzer-store region -verify %s

class A {
public:
  A();
  ~A() {};
  
  virtual int foo() = 0;
  virtual void bar() = 0;
  void f() {
    foo(); // expected-warning{{Call pure virtual functions during construction or destruction may leads undefined behaviour}}
  }
};

class B : public A {
public:
  B() {
    foo(); // expected-warning{{Call virtual functions during construction or destruction will never go to a more derived class}}
  }
  ~B();
  
  virtual int foo();
  virtual void bar() { foo(); }  // expected-warning{{Call virtual functions during construction or destruction will never go to a more derived class}}
};

A::A() {
  f();
}

B::~B() {
  this->B::foo(); // no-warning
  this->B::bar();
  this->foo(); // expected-warning{{Call virtual functions during construction or destruction will never go to a more derived class}}
}

class C : public B {
public:
  C();
  ~C();
  
  virtual int foo();
  void f(int i);
};

C::C() {
  f(foo()); // expected-warning{{Call virtual functions during construction or destruction will never go to a more derived class}}
}

int main() {
  A *a;
  B *b;
  C *c;
}

#include "virtualcall.h"

#define AS_SYSTEM
#include "virtualcall.h"
#undef AS_SYSTEM
