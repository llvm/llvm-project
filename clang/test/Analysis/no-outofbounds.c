// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core,alpha.unix,alpha.security.ArrayBound -analyzer-store=region -verify %s

//===----------------------------------------------------------------------===//
// This file tests cases where we should not flag out-of-bounds warnings.
//===----------------------------------------------------------------------===//

void f() {
  long x = 0;
  char *y = (char*) &x;
  char c = y[0] + y[1] + y[2]; // no-warning
  short *z = (short*) &x;
  short s = z[0] + z[1]; // no-warning
}

void g() {
  int a[2];
  char *b = (char*)a;
  b[3] = 'c'; // no-warning
}

typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

void field() {
  struct vec { size_t len; int data[0]; };
  // FIXME: Not warn for this.
  struct vec *a = malloc(sizeof(struct vec) + 10); // expected-warning {{Cast a region whose size is not a multiple of the destination type size}}
  a->len = 10;
  a->data[1] = 5; // no-warning
  free(a);
}
