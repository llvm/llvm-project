// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if we search for clones in functions.

void log();

int max(int a, int b) { // expected-warning{{Detected code clone.}}
  log();
  if (a > b)
    return a;
  return b;
}

int maxClone(int x, int y) { // expected-note{{Related code clone is here.}}
  log();
  if (x > y)
    return x;
  return y;
}

// Functions below are not clones and should not be reported.

int foo(int a, int b) { // no-warning
  return a + b;
}
