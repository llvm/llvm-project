// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -DCL20 -cl-std=CL2.0

static constant int G1 = 0;
int G2 = 0;
global int G3 = 0;
local int G4 = 0;// expected-error{{program scope variable must reside in global or constant address space}}

void kernel foo() {
  static int S1 = 5;
  static global int S2 = 5;
  static private int S3 = 5;// expected-error{{program scope variable must reside in global or constant address space}}

  constant int L1 = 0;
  local int L2;
}
