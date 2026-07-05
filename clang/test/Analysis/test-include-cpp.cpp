// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

#include "test-include-cpp.h"

int TestIncludeClass::test1(int *p) {
  p = 0;
  return *p; // expected-warning{{Dereference of null pointer}}
}

int TestIncludeClass::test2(int *p) {
  p = 0;
  return *p; // expected-warning{{Dereference of null pointer}}
}
