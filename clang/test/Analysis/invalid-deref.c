// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

typedef unsigned uintptr_t;

void f1(void) {
  int *p;
  *p = 0; // expected-warning{{Dereference of undefined pointer value}}
}

struct foo_struct {
  int x;
};

int f2(void) {
  struct foo_struct *p;

  return p->x++; // expected-warning{{Access to field 'x' results in a dereference of an undefined pointer value (loaded from variable 'p')}}
}

int f3(void) {
  char *x;
  int i = 2;

  return x[i + 1]; // expected-warning{{Array access (from variable 'x') results in an undefined pointer dereference}}
}

int f3_b(void) {
  char *x;
  int i = 2;

  return x[i + 1]++; // expected-warning{{Array access (from variable 'x') results in an undefined pointer dereference}}
}
