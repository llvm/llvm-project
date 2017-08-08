// RUN: %clang_scudo -fsized-deallocation %s -o %t
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=1     %run %t gooddel    2>&1
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=1 not %run %t baddel     2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=0     %run %t baddel     2>&1
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=1     %run %t gooddelarr 2>&1
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=1 not %run %t baddelarr  2>&1 | FileCheck %s
// RUN: SCUDO_OPTIONS=DeleteSizeMismatch=0     %run %t baddelarr  2>&1

// Ensures that the sized delete operator errors out when the appropriate
// option is passed and the sizes do not match between allocation and
// deallocation functions.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <new>

int main(int argc, char **argv)
{
  assert(argc == 2);
  if (!strcmp(argv[1], "gooddel")) {
    long long *p = new long long;
    operator delete(p, sizeof(long long));
  }
  if (!strcmp(argv[1], "baddel")) {
    long long *p = new long long;
    operator delete(p, 2);
  }
  if (!strcmp(argv[1], "gooddelarr")) {
    char *p = new char[64];
    operator delete[](p, 64);
  }
  if (!strcmp(argv[1], "baddelarr")) {
    char *p = new char[63];
    operator delete[](p, 64);
  }
  return 0;
}

// CHECK: ERROR: invalid sized delete on chunk at address
