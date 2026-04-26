// RUN: %clangxx_tysan -O0 %s -c -o %t.o
// RUN: %clangxx_tysan -O0 %s -DPMAIN -c -o %tm.o
// RUN: %clangxx_tysan -O0 %s -DPINIT -c -o %tinit.o
// RUN: %clangxx_tysan -O0 %t.o %tm.o %tinit.o -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

extern "C" {
typedef struct X {
  int *start;
  int *end;
  int i;
} X;
};

#ifdef PMAIN
int foo(struct X *);
void bar(struct X *);
void init(struct X *);

int main() {
  struct X x;
  init(&x);
  printf("%d\n", foo(&x));
  free(x.start);
  return 0;
}

#elif PINIT

void init(struct X *x) {
  x->start = (int *)calloc(100, sizeof(int));
  x->end = x->start + 99;
  x->i = 0;
}

#else

__attribute__((noinline)) int foo(struct X *x) {
  if (x->start < x->end)
    return 30;
  return 10;
}

void bar(struct X *x) { x->end = NULL; }

#endif

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
