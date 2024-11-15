// RUN: %clang_analyze_cc1 -analyzer-checker=security.PointerSub -analyzer-output=text-minimal -verify %s

void f1(void) {
  int x, y, z[10];
  int d = &y - &x; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = z - &y; // expected-warning{{Subtraction of two pointers that do not point into the same array is undefined behavior}}
  d = &x - &x; // no-warning (subtraction of any two identical pointers is allowed)
  d = (long *)&x - (long *)&x;
  d = (&x + 1) - &x; // no-warning ('&x' is like a single-element array)
  d = &x - (&x + 1); // no-warning
  d = (&x + 0) - &x; // no-warning
  d = (z + 10) - z; // no-warning
}

void f2(void) {
  int a[10], b[10], c; // expected-note{{Array at the left-hand side of subtraction}} \
                       // expected-note2{{Array at the right-hand side of subtraction}}
  int *p = &a[2];
  int *q = &a[8];
  int d = q - p; // no-warning (pointers into the same array)

  q = &b[3];
  d = q - p; // expected-warning{{Subtraction of two pointers that}}

  d = &a[4] - a; // no-warning
  d = &a[2] - p; // no-warning
  d = &c - p; // expected-warning{{Subtraction of two pointers that}}

  d = (int *)((char *)(&a[4]) + sizeof(int)) - &a[4]; // no-warning (pointers into the same array data)
  d = (int *)((char *)(&a[4]) + 1) - &a[4]; // expected-warning{{Subtraction of two pointers that}}
}

void f3(void) {
  int a[3][4]; // expected-note{{Array at the left-hand side of subtraction}} \
               // expected-note2{{Array at the right-hand side of subtraction}}
  int d;

  d = &(a[2]) - &(a[1]);
  d = a[2] - a[1]; // expected-warning{{Subtraction of two pointers that}}
  d = a[1] - a[1];
  d = &(a[1][2]) - &(a[1][0]);
  d = &(a[1][2]) - &(a[0][0]); // expected-warning{{Subtraction of two pointers that}}

  d = (int *)((char *)(&a[2][2]) + sizeof(int)) - &a[2][2]; // expected-warning{{Subtraction of two pointers that}}
  d = (int *)((char *)(&a[2][2]) + 1) - &a[2][2]; // expected-warning{{Subtraction of two pointers that}}
  d = (int (*)[4])((char *)&a[2] + sizeof(int (*)[4])) - &a[2]; // expected-warning{{Subtraction of two pointers that}}
  d = (int (*)[4])((char *)&a[2] + 1) - &a[2]; // expected-warning{{Subtraction of two pointers that}}
}

void f4(void) {
  int n = 4, m = 3;
  int a[n][m];
  int (*p)[m] = a; // p == &a[0]
  p += 1; // p == &a[1]

  // FIXME: This is a known problem with -Wpointer-arith (https://github.com/llvm/llvm-project/issues/28328)
  int d = p - a; // d == 1 // expected-warning{{subtraction of pointers to type 'int[m]' of zero size has undefined behavior}}

  // FIXME: This is a known problem with -Wpointer-arith (https://github.com/llvm/llvm-project/issues/28328)
  d = &(a[2]) - &(a[1]); // expected-warning{{subtraction of pointers to type 'int[m]' of zero size has undefined behavior}}

  d = a[2] - a[1]; // expected-warning{{Subtraction of two pointers that}}
}

struct S {
  int a;
  int b;
  int c[10]; // expected-note2{{Array at the right-hand side of subtraction}}
  int d[10]; // expected-note2{{Array at the left-hand side of subtraction}}
};

void f5(void) {
  struct S s;
  int y;
  int d;

  d = &s.b - &s.a; // expected-warning{{Subtraction of two pointers that}}
  d = &s.c[0] - &s.a; // expected-warning{{Subtraction of two pointers that}}
  d = &s.b - &y; // expected-warning{{Subtraction of two pointers that}}
  d = &s.c[3] - &s.c[2];
  d = &s.d[3] - &s.c[2]; // expected-warning{{Subtraction of two pointers that}}
  d = s.d - s.c; // expected-warning{{Subtraction of two pointers that}}

  struct S sa[10];
  d = &sa[2] - &sa[1];
  d = &sa[2].a - &sa[1].b; // expected-warning{{Subtraction of two pointers that}}
}

void f6(void) {
  long long l = 2;
  char *a1 = (char *)&l;
  int d = a1[3] - l;

  long long la1[3] = {1}; // expected-note{{Array at the right-hand side of subtraction}}
  long long la2[3] = {1}; // expected-note{{Array at the left-hand side of subtraction}}
  char *pla1 = (char *)la1;
  char *pla2 = (char *)la2;
  d = pla1[1] - pla1[0];
  d = (long long *)&pla1[1] - &l; // expected-warning{{Subtraction of two pointers that}}
  d = &pla2[3] - &pla1[3]; // expected-warning{{Subtraction of two pointers that}}
}

void f7(int *p) {
  int a[10];
  int d = &a[10] - p; // no-warning ('p' is unknown, even if it cannot point into 'a')
}

void f8(int n) {
  int a[10] = {1};
  int d = a[n] - a[0]; // no-warning
}

int f9(const char *p1) {
  const char *p2 = p1;
  --p1;
  ++p2;
  return p1 - p2; // no-warning
}

int f10(struct S *p1, struct S *p2) {
  return &p1->c[5] - &p2->c[5]; // no-warning
}

struct S1 {
  int a;
  int b; // expected-note{{Object at the right-hand side of subtraction}}
};

int f11() {
  struct S1 s; // expected-note{{Object at the left-hand side of subtraction}}
  return (char *)&s - (char *)&s.b; // expected-warning{{Subtraction of two pointers that}}
}

struct S2 {
  char *p1;
  char *p2;
};

void init_S2(struct S2 *);

int f12() {
  struct S2 s;
  init_S2(&s);
  return s.p1 - s.p2; // no-warning (pointers are unknown)
}
