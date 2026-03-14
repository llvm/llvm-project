// Regression check that -pedantic-errors doesn't cause other diagnostics to
// become errors.
//
// RUN: %clang_cc1 -verify -Weverything -pedantic-errors %s

int f0(int, unsigned);
int f0(int x, unsigned y) {
  return x < y; // expected-warning {{comparison of integers}}
}
