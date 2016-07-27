// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if sub-sequences can match with normal sequences.

void log2(int a);
void log();

int max(int a, int b) {
  log2(a);
  log(); // expected-warning{{Detected code clone.}}
  if (a > b)
    return a;
  return b;
}

int maxClone(int a, int b) {
  log(); // expected-note{{Related code clone is here.}}
  if (a > b)
    return a;
  return b;
}

// Functions below are not clones and should not be reported.

int foo(int a, int b) { // no-warning
  return a + b;
}
