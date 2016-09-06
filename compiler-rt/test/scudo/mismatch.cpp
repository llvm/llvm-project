// RUN: %clang_scudo %s -o %t
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=1 not %run %t mallocdel   2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=0     %run %t mallocdel   2>&1
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=1 not %run %t newfree     2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=0     %run %t newfree     2>&1
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=1 not %run %t memaligndel 2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=DeallocationTypeMismatch=0     %run %t memaligndel 2>&1

// Tests that type mismatches between allocation and deallocation functions are
// caught when the related option is set.

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "mallocdel")) {
    int *p = (int *)malloc(16);
    if (!p)
      return 1;
    delete p;
  }
  if (!strcmp(argv[1], "newfree")) {
    int *p = new int;
    if (!p)
      return 1;
    free((void *)p);
  }
  if (!strcmp(argv[1], "memaligndel")) {
    int *p = (int *)memalign(0x10, 0x10);
    if (!p)
      return 1;
    delete p;
  }
  return 0;
}

// CHECK: ERROR: allocation type mismatch on address
