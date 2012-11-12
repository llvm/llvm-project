// RUN: %clang_cc1 -analyze -analyzer-checker=unix.MallocSizeof -verify %s

#include <stddef.h>

void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

struct A {};
struct B {};

void foo(unsigned int unsignedInt, unsigned int readSize) {
  int *ip1 = malloc(sizeof(1));
  int *ip2 = malloc(4 * sizeof(int));

  long *lp1 = malloc(sizeof(short)); // expected-warning {{Result of 'malloc' is converted to a pointer of type 'long', which is incompatible with sizeof operand type 'short'}}
  long *lp2 = malloc(5 * sizeof(double)); // expected-warning {{Result of 'malloc' is converted to a pointer of type 'long', which is incompatible with sizeof operand type 'double'}}
  char *cp3 = malloc(5 * sizeof(char) + 2); // no warning
  unsigned char *buf = malloc(readSize + sizeof(unsignedInt)); // no warning

  struct A *ap1 = calloc(1, sizeof(struct A));
  struct A *ap2 = calloc(2, sizeof(*ap1));
  struct A *ap3 = calloc(2, sizeof(ap1)); // expected-warning {{Result of 'calloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'struct A *'}}
  struct A *ap4 = calloc(3, sizeof(struct A*)); // expected-warning {{Result of 'calloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'struct A *'}}
  struct A *ap5 = calloc(4, sizeof(struct B)); // expected-warning {{Result of 'calloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'struct B'}}
  struct A *ap6 = realloc(ap5, sizeof(struct A));
  struct A *ap7 = realloc(ap5, sizeof(struct B)); // expected-warning {{Result of 'realloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'struct B'}}
}

// Don't warn when the types differ only by constness.
void ignore_const() {
  const char **x = (const char **)malloc(1 * sizeof(char *)); // no-warning
  const char ***y = (const char ***)malloc(1 * sizeof(char *)); // expected-warning {{Result of 'malloc' is converted to a pointer of type 'const char **', which is incompatible with sizeof operand type 'char *'}}
  free(x);
}

int *mallocArraySize() {
  static const int sTable[10];
  static const int nestedTable[10][2];
  int *table = malloc(sizeof sTable);
  int *table1 = malloc(sizeof nestedTable);
  int (*table2)[2] = malloc(sizeof nestedTable);
  int (*table3)[10][2] = malloc(sizeof nestedTable);
  return table;
}

int *mallocWrongArraySize() {
  static const double sTable[10];
  int *table = malloc(sizeof sTable); // expected-warning {{Result of 'malloc' is converted to a pointer of type 'int', which is incompatible with sizeof operand type 'const double [10]'}}
  return table;
}
