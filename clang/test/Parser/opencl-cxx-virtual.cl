// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -fsyntax-only -verify

// Test that virtual functions and abstract classes are rejected.
class virtual_functions {
  virtual void bad1() {}
  //expected-error@-1 {{virtual functions are not supported in OpenCL C++}}

  virtual void bad2() = 0;
  //expected-error@-1 {{virtual functions are not supported in OpenCL C++}}
  //expected-error@-2 {{'bad2' is not virtual and cannot be declared pure}}
};

template <typename T>
class X {
  virtual T f();
  //expected-error@-1 {{virtual functions are not supported in OpenCL C++}}
};

// Test that virtual base classes are allowed.
struct A {
  int a;
  void foo();
};

struct B : virtual A {
  int b;
};

struct C : public virtual A {
  int c;
};

struct D : B, C {
  int d;
};

kernel void virtual_inheritance() {
  D d;

  d.foo();
  d.a = 11;
  d.b = 22;
  d.c = 33;
  d.d = 44;
}
