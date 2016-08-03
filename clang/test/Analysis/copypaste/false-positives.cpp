// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// This test contains false-positive reports from the CloneChecker that need to
// be fixed.

void log();

int max(int a, int b) { // expected-warning{{Detected code clone.}}
  log();
  if (a > b)
    return a;
  return b;
}

// FIXME: Detect different variable patterns.
int min2(int a, int b) { // expected-note{{Related code clone is here.}}
  log();
  if (b > a)
    return a;
  return b;
}
