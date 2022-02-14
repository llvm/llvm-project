// RUN: %clang_analyze_cc1 -triple x86_64-unknown-freebsd %s

#include "Inputs/system-header-simulator.h"

#define M_ZERO 0x0100
#define NULL ((void *)0)

void *malloc(size_t, void *, int);

struct test {
};

void foo(struct test *);

void test_zeroed(void) {
  struct test **list, *t;
  int i;

  list = malloc(sizeof(*list) * 10, NULL, M_ZERO);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i];
    foo(t);
  }
  free(list); // no-warning
}

void test_nonzero(void) {
  struct test **list, *t;
  int i;

  list = malloc(sizeof(*list) * 10, NULL, 0);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i]; // expected-warning{{undefined}}
    foo(t);
  }
  free(list);
}

void test_indeterminate(int flags) {
  struct test **list, *t;
  int i;

  list = malloc(sizeof(*list) * 10, NULL, flags);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i]; // expected-warning{{undefined}}
    foo(t);
  }
  free(list);
}
