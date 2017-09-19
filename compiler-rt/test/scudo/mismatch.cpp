// RUN: %clang_scudo %s -o %t
// RUN: %env_scudo_opts=DeallocationTypeMismatch=1 not %run %t mallocdel       2>&1 | FileCheck --check-prefix=CHECK-dealloc %s
// RUN: %env_scudo_opts=DeallocationTypeMismatch=0     %run %t mallocdel       2>&1
// RUN: %env_scudo_opts=DeallocationTypeMismatch=1 not %run %t newfree         2>&1 | FileCheck --check-prefix=CHECK-dealloc %s
// RUN: %env_scudo_opts=DeallocationTypeMismatch=0     %run %t newfree         2>&1
// RUN: %env_scudo_opts=DeallocationTypeMismatch=1 not %run %t memaligndel     2>&1 | FileCheck --check-prefix=CHECK-dealloc %s
// RUN: %env_scudo_opts=DeallocationTypeMismatch=0     %run %t memaligndel     2>&1
// RUN: %env_scudo_opts=DeallocationTypeMismatch=1 not %run %t memalignrealloc 2>&1 | FileCheck --check-prefix=CHECK-realloc %s
// RUN: %env_scudo_opts=DeallocationTypeMismatch=0     %run %t memalignrealloc 2>&1

// Tests that type mismatches between allocation and deallocation functions are
// caught when the related option is set.

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "mallocdel")) {
    int *p = (int *)malloc(16);
    assert(p);
    delete p;
  }
  if (!strcmp(argv[1], "newfree")) {
    int *p = new int;
    assert(p);
    free((void *)p);
  }
  if (!strcmp(argv[1], "memaligndel")) {
    int *p = (int *)memalign(16, 16);
    assert(p);
    delete p;
  }
  if (!strcmp(argv[1], "memalignrealloc")) {
    void *p = memalign(16, 16);
    assert(p);
    p = realloc(p, 32);
    free(p);
  }
  return 0;
}

// CHECK-dealloc: ERROR: allocation type mismatch when deallocating address
// CHECK-realloc: ERROR: allocation type mismatch when reallocating address
