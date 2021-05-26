// Test various -fsanitize= additional flags combinations.

// RUN: %clang_scudo %s -o %t
// RUN: not --crash %run %t 2>&1 | FileCheck %s

// RUN: %clang_scudo -shared-libsan %s -o %t
// RUN: env LD_LIBRARY_PATH=`dirname %shared_libscudo`:$LD_LIBRARY_PATH not --crash %run %t 2>&1 | FileCheck %s

// RUN: %clang_scudo -static-libsan %s -o %t
// RUN: not --crash %run %t 2>&1 | FileCheck %s
// RUN: %clang_scudo -static-libsan -fsanitize-minimal-runtime %s -o %t
// RUN: not --crash %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  unsigned long *p = (unsigned long *)malloc(sizeof(unsigned long));
  assert(p);
  *p = 0;
  free(p);
  free(p);
  return 0;
}

// CHECK: ERROR: invalid chunk state
