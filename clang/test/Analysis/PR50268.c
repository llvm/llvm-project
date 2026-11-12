// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -verify %s \
// RUN:    -analyzer-config eagerly-assume=true

// expected-no-diagnostics


int test(unsigned long a, unsigned long c, int b) {
  c -= a;
  if (0 >= b) {}
  c == b;
  return c ? 0 : 2; // no-crash
}
