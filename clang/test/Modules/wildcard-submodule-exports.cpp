// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodule-cache-path %t -fmodules -I %S/Inputs/wildcard-submodule-exports %s -verify
// FIXME: When we have a syntax for modules in C++, use that.

@__experimental_modules_import C.One;

void test_C_One() {
  int *A1_ptr = A1;
  long *C1_ptr = C1;
  (void)B1; // expected-error{{use of undeclared identifier 'B1'}}
}

@__experimental_modules_import C.Two;

void test_C_Two() {
  unsigned int *A2_ptr = A2;
  unsigned short *B2_ptr = B2;
  unsigned long *C2_ptr = C2;
}

@__experimental_modules_import B.One;

void test_B_One() {
  short *B1_ptr = B1;
}

