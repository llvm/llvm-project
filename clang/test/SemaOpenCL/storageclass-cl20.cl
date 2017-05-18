// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

static constant int G1 = 0;
int G2 = 0;
global int G3 = 0;
local int G4 = 0;              // expected-error{{program scope variable must reside in global or constant address space}}

void kernel foo() {
  static int S1 = 5;
  static global int S2 = 5;
  static private int S3 = 5;   // expected-error{{static local variable must reside in global or constant address space}}

  constant int L1 = 0;
  local int L2;
  global int L3;                              // expected-error{{function scope variable cannot be declared in global address space}}
  generic int L4;                             // expected-error{{automatic variable qualified with an invalid address space}}
  __attribute__((address_space(100))) int L5; // expected-error{{automatic variable qualified with an invalid address space}}

  extern global int G5;
  extern int G6; // expected-error{{extern variable must reside in global or constant address space}}
}
