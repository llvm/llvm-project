// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux %s

#define __GFP_ZERO 0x8000
#define NULL ((void *)0)

typedef __typeof(sizeof(int)) size_t;

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
  kfree(list); // no-warning
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
  kfree(list);
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
  kfree(list);
}

typedef unsigned long long uint64_t;

struct malloc_type;

void *malloc(unsigned long size, struct malloc_type *mtp, int flags);

void test_3arg_malloc(struct malloc_type *mtp) {
  struct test **list, *t;
  int i;

  list = malloc(sizeof(*list) * 10, mtp, __GFP_ZERO);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i];
    foo(t);
  }
  kfree(list); // no-warning
}

void test_3arg_malloc_nonzero(struct malloc_type *mtp) {
  struct test **list, *t;
  int i;

  list = malloc(sizeof(*list) * 10, mtp, 0);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i]; // expected-warning{{undefined}}
    foo(t);
  }
  kfree(list);
}

void test_3arg_malloc_indeterminate(struct malloc_type *mtp, int flags) {
  struct test **list, *t;
  int i;

  list = alloc(sizeof(*list) * 10, mtp, flags);
  if (list == NULL)
    return;

  for (i = 0; i < 10; i++) {
    t = list[i]; // expected-warning{{undefined}}
    foo(t);
  }
  kfree(list);
}
