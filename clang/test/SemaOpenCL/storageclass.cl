// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

static constant int G1 = 0;
constant int G2 = 0;
int G3 = 0;        // expected-error{{program scope variable must reside in constant address space}}
global int G4 = 0; // expected-error{{program scope variable must reside in constant address space}}

void kernel foo(int x) {
  // static is not allowed at local scope before CL2.0
  static int S1 = 5;          // expected-error{{variables in function scope cannot be declared static}}
  static constant int S2 = 5; // expected-error{{variables in function scope cannot be declared static}}

  constant int L1 = 0;
  local int L2;

  auto int L3 = 7; // expected-error{{OpenCL version 1.2 does not support the 'auto' storage class specifier}}
  global int L4;   // expected-error{{function scope variable cannot be declared in global address space}}

  constant int L5 = x; // expected-error {{initializer element is not a compile-time constant}}
  global int *constant L6 = &G4;
  private int *constant L7 = &x; // expected-error {{initializer element is not a compile-time constant}}
  constant int *constant L8 = &L1;
  local int *constant L9 = &L2; // expected-error {{initializer element is not a compile-time constant}}
}

static void kernel bar() { // expected-error{{kernel functions cannot be declared static}}
}

void f() {
  constant int L1 = 0; // expected-error{{non-kernel function variable cannot be declared in constant address space}}
  local int L2;        // expected-error{{non-kernel function variable cannot be declared in local address space}}
  {
    constant int L1 = 0; // expected-error{{non-kernel function variable cannot be declared in constant address space}}
    local int L2;        // expected-error{{non-kernel function variable cannot be declared in local address space}}
  }
  global int L3; // expected-error{{function scope variable cannot be declared in global address space}}
  extern constant float L4;
  extern local float L5; // expected-error{{extern variable must reside in constant address space}}
  static int L6 = 0;     // expected-error{{variables in function scope cannot be declared static}}
  static int L7;         // expected-error{{variables in function scope cannot be declared static}}
}
