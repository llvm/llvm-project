// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -verify %s
// expected-no-diagnostics

int f1() {
  int x = 0, y = 1;
  return x += y; // Should bind a location to 'x += y'. No crash.
}
