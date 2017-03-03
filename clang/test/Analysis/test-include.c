// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

#include "test-include.h"
#define DIVYX(X,Y) Y/X

void test_01(int *data) {
  data = 0;
  *data = 1; // expected-warning{{Dereference of null pointer}}
}

int test_02() {
  int res = DIVXY(1,0); // expected-warning{{Division by zero}}
                        // expected-warning@-1{{division by zero is undefined}}
  return res;
}

int test_03() {
  int res = DIVYX(0,1); // expected-warning{{Division by zero}}
                        // expected-warning@-1{{division by zero is undefined}}
  return res;
}