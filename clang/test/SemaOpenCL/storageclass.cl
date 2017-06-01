// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

static constant int G1 = 0;
constant int G2 = 0;
int G3 = 0;        // expected-error{{program scope variable must reside in constant address space}}
global int G4 = 0; // expected-error{{program scope variable must reside in constant address space}}

void kernel foo() {
  // static is not allowed at local scope before CL2.0
  static int S1 = 5;          // expected-error{{variables in function scope cannot be declared static}}
  static constant int S2 = 5; // expected-error{{variables in function scope cannot be declared static}}

  constant int L1 = 0;
  local int L2;

  auto int L3 = 7; // expected-error{{OpenCL version 1.2 does not support the 'auto' storage class specifier}}
  global int L4;   // expected-error{{function scope variable cannot be declared in global address space}}
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
}
