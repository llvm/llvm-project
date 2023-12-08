// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

int f1() {
  int x = 0, y = 1;
  return x += y; // Should bind a location to 'x += y'. No crash.
}
