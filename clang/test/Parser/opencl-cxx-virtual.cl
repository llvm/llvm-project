// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -fsyntax-only -verify -DFUNCPTREXT

#ifdef FUNCPTREXT
#pragma OPENCL EXTENSION __cl_clang_function_pointers : enable
//expected-no-diagnostics
#endif

// Test that virtual functions and abstract classes are rejected
// unless specific clang extension is used.
class virtual_functions {
  virtual void bad1() {}
#ifndef FUNCPTREXT
  //expected-error@-2 {{virtual functions are not supported in C++ for OpenCL}}
#endif

  virtual void bad2() = 0;
#ifndef FUNCPTREXT
  //expected-error@-2 {{virtual functions are not supported in C++ for OpenCL}}
  //expected-error@-3 {{'bad2' is not virtual and cannot be declared pure}}
#endif
};

template <typename T>
class X {
  virtual T f();
#ifndef FUNCPTREXT
  //expected-error@-2 {{virtual functions are not supported in C++ for OpenCL}}
#endif
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
