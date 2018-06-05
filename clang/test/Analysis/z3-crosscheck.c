// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -DNO_CROSSCHECK -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config crosscheck-with-z3=true -verify %s
// REQUIRES: z3

int foo(int x) 
{
  int *z = 0;
  if ((x & 1) && ((x & 1) ^ 1))
#ifdef NO_CROSSCHECK
      return *z; // expected-warning {{Dereference of null pointer (loaded from variable 'z')}}
#else
      return *z; // no-warning
#endif
  return 0;
}

void g(int d);

void f(int *a, int *b) {
  int c = 5;
  if ((a - b) == 0)
    c = 0;
  if (a != b)
#ifdef NO_CROSSCHECK
    g(3 / c); // expected-warning {{Division by zero}}
#else
    g(3 / c); // no-warning
#endif
}

_Bool nondet_bool();

void h(int d) {
  int x, y, k, z = 1;
#ifdef NO_CROSSCHECK
  while (z < k) { // expected-warning {{The right operand of '<' is a garbage value}}
#else
  while (z < k) { // expected-warning {{The right operand of '<' is a garbage value}}
#endif
    z = 2 * z;
  }
}

void i() {
  _Bool c = nondet_bool();
  if (c) {
    h(1);
  } else {
    h(2);
  }
}
