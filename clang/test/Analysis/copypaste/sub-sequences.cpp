// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This tests if sub-sequences can match with normal sequences.

void log2(int a);
void log();

int max(int a, int b) {
  log2(a);
  log(); // expected-warning{{Duplicate code detected}}
  if (a > b)
    return a;
  return b;
}

int maxClone(int a, int b) {
  log(); // expected-note{{Similar code here}}
  if (a > b)
    return a;
  return b;
}

// Functions below are not clones and should not be reported.

int foo(int a, int b) { // no-warning
  return a + b;
}
