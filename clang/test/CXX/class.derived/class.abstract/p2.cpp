// RUN: %clang_cc1 -std=c++1z -verify %s

// no objects of an abstract class can be created except as subobjects of a
// class derived from it

struct A {
  A() {}
  A(int) : A() {} // ok

  virtual void f() = 0; // expected-note 1+{{unimplemented}}
};

void f(A &&a);

void g() {
  f({}); // expected-error {{abstract class}}
  f({0}); // expected-error {{abstract class}}
  f(0); // expected-error {{abstract class}}
}

struct B : A {
  B() : A() {} // ok
};
