// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -error-on-deserialized-decl S1_method -include-pch %t -analyze -analyzer-checker=core %s
// RUN: %clang_cc1 -include-pch %t -analyze -analyzer-checker=core -verify %s

#ifndef HEADER
#define HEADER
// Header.

void S1_method(); // This should not be deserialized.


#else
// Using the header.

int test() {
  int x = 0;
  return 5/x; //expected-warning {{Division by zero}}
}

#endif
