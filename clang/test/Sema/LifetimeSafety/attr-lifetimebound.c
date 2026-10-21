// RUN: %clang_cc1 -std=c99 -verify %s

int *f(int* p __attribute__((lifetimebound)));

int *g() {
  int i;
  return f(&i); // expected-warning {{address of stack memory associated with local variable 'i' returned}}
}
