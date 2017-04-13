// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux %s

#include "Inputs/system-header-simulator.h"

#define __GFP_ZERO 0x8000
#define NULL ((void *)0)

void *kmalloc(size_t, int);

struct test {
};

void foo(struct test *);

void test_zeroed() {
  struct test **list, *t;
  int i;

  list = kmalloc(sizeof(*list) * 10, __GFP_ZERO);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i];
    foo(t);
  }
  free(list); // no-warning
}

void test_nonzero() {
  struct test **list, *t;
  int i;

  list = kmalloc(sizeof(*list) * 10, 0);
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

  list = kmalloc(sizeof(*list) * 10, flags);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i]; // expected-warning{{undefined}}
    foo(t);
  }
  free(list);
}
