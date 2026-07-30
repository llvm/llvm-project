// RUN: %clang_cc1 -fsyntax-only -Wshift-count-negative -fblocks -verify %s

int f(int a) {
  const int i = -1;
  return a << i; // expected-warning{{shift count is negative}}
}
