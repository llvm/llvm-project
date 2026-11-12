// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/wildcard-submodule-exports %s -verify
// FIXME: When we have a syntax for modules in C++, use that.

@import C.One;

void test_C_One() {
  int *A1_ptr = A1;
  long *C1_ptr = C1;
  (void)B1; // expected-error{{use of undeclared identifier 'B1'}}
}

@import C.Two;

void test_C_Two() {
  unsigned int *A2_ptr = A2;
  unsigned short *B2_ptr = B2;
  unsigned long *C2_ptr = C2;
}

@import B.One;

void test_B_One() {
  short *B1_ptr = B1;
}

