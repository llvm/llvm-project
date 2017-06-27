// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core,unix,alpha.unix -std=gnu99 -analyzer-store=region -verify %s

#include "Inputs/system-header-simulator.h"

typedef __typeof(sizeof(int)) size_t;
void *memset(void *__s, int __c, size_t __n);
void *malloc(size_t __size);
void free(void *__ptr);

// The store for 'a[1]' should not be removed mistakenly. SymbolicRegions may
// also be live roots.
void f14(int *a) {
  int i;
  a[1] = 1;
  i = a[1];
  if (i != 1) {
    int *p = 0;
    i = *p; // no-warning
  }
}

void foo() {
  int *x = malloc(sizeof(int));
  memset(x, 0, sizeof(int));
  int n = 1 / *x; // FIXME: no-warning
  free(x);
}

void bar() {
  int *x = malloc(sizeof(int));
  memset(x, 0, 1);
  int n = 1 / *x; // no-warning
  free(x);
}

void testConcreteNull() {
  int *x = 0;
  memset(x, 0, 1); // expected-warning {{Null pointer argument in call to memory set function}}
}

void testStackArray() {
  char buf[13];
  memset(buf, 0, 1); // no-warning
}

void testHeapSymbol() {
  char *buf = (char *)malloc(13);
  memset(buf, 0, 1); // no-warning
  free(buf);
}

void testStackArrayOutOfBound() {
  char buf[1];
  memset(buf, 0, 1024); // expected-warning {{Memory set function accesses out-of-bound array element}}
}

void testHeapSymbolOutOfBound() {
  char *buf = (char *)malloc(1);
  memset(buf, 0, 1024); // expected-warning {{Memory set function accesses out-of-bound array element}}
  free(buf);
}

void testStackArraySameSize() {
  char buf[1];
  memset(buf, 0, sizeof(buf)); // no-warning
}

void testHeapSymbolSameSize() {
  char *buf = (char *)malloc(1);
  memset(buf, 0, 1); // no-warning
  free(buf);
}
