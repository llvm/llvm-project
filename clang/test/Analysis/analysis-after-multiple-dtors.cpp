// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

#include "Inputs/system-header-simulator-cxx.h"

struct Test {
  Test() {}
  ~Test();
};

int foo() {
  struct a {
  // The dtor invocation of 'b' and 'c' used to create
  // a loop in the egraph and the analysis stopped after
  // this point.
    Test b, c;
  } d;
  return 1;
}

int main() {
  if (foo()) {
  }

  int x;
  int y = x;
  // expected-warning@-1{{Assigned value is garbage or undefined}}
  (void)y;
}
