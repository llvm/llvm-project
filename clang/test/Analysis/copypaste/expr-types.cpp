// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics


int foo1(int a, int b) {
  if (a > b)
    return a;
  return b;
}

// Different types, so not a clone
int foo2(long a, long b) {
  if (a > b)
    return a;
  return b;
}
