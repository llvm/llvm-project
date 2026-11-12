// RUN: %clang_cc1 -verify -Wno-everything -Wsign-compare %s

int f0(int, unsigned);
int f0(int x, unsigned y) {
  if (x=3);
  return x < y; // expected-warning {{comparison of integers}}
}
