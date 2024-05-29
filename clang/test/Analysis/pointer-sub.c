// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.PointerSub -verify %s

void f1(void) {
  int x, y, z[10];
  int d = &y - &x; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = z - &y; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = &x - &x; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = (long*)&x - (long*)&x;
}

void f2(void) {
  int a[10], b[10], c;
  int *p = &a[2];
  int *q = &a[8];
  int d = q - p; // no-warning

  q = &b[3];
  d = q - p; // expected-warning{{Subtraction of two pointers that}}

  q = a + 10;
  d = q - p; // no warning (use of pointer to one after the end is allowed)
  d = &a[4] - a; // no warning

  q = a + 11;
  d = q - a; // ?

  d = &c - p; // expected-warning{{Subtraction of two pointers that}}
}

void f3(void) {
  int a[3][4];
  int d;

  d = &(a[2]) - &(a[1]);
  d = a[2] - a[1]; // expected-warning{{Subtraction of two pointers that}}
  d = a[1] - a[1];
  d = &(a[1][2]) - &(a[1][0]);
  d = &(a[1][2]) - &(a[0][0]); // expected-warning{{Subtraction of two pointers that}}
}

void f4(void) {
  int n = 4, m = 3;
  int a[n][m];
  int (*p)[m] = a; // p == &a[0]
  p += 1; // p == &a[1]
  int d = p - a; // d == 1 // expected-warning{{subtraction of pointers to type 'int[m]' of zero size has undefined behavior}}

  d = &(a[2]) - &(a[1]); // expected-warning{{subtraction of pointers to type 'int[m]' of zero size has undefined behavior}}
  d = a[2] - a[1]; // expected-warning{{Subtraction of two pointers that}}
}

typedef struct {
  int a;
  int b;
  int c[10];
  int d[10];
} S;

void f5(void) {
  S s;
  int y;
  int d;

  d = &s.b - &s.a; // expected-warning{{Subtraction of two pointers that}}
  d = &s.c[0] - &s.a; // expected-warning{{Subtraction of two pointers that}}
  d = &s.b - &y; // expected-warning{{Subtraction of two pointers that}}
  d = &s.c[3] - &s.c[2];
  d = &s.d[3] - &s.c[2]; // expected-warning{{Subtraction of two pointers that}}
  d = s.d - s.c; // expected-warning{{Subtraction of two pointers that}}

  S sa[10];
  d = &sa[2] - &sa[1];
  d = &sa[2].a - &sa[1].b; // expected-warning{{Subtraction of two pointers that}}
}
