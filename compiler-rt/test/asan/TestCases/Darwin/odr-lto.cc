// Check that -asan-use-private-alias silence the false
// positive ODR violation on Darwin with LTO.

// REQUIRES: lto

// RUN: %clangxx_asan -DPART=0 -c %s -o %t-1.o -flto -mllvm -asan-use-private-alias
// RUN: %clangxx_asan -DPART=1 -c %s -o %t-2.o -flto -mllvm -asan-use-private-alias
// RUN: %clangxx_asan %t-1.o %t-2.o -o %t -flto
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
void putstest();

#if PART == 1

static const char *my_global = "test\n\00abc";

int main()
{
  fputs(my_global, stderr);
  putstest();
  fprintf(stderr, "Done.\n");
  return 0;
}

#else // PART == 1

static const char *my_other_global = "test\n\00abc";

void putstest()
{
  fputs(my_other_global, stderr);
}

#endif // PART == 1

// CHECK-NOT: ERROR: AddressSanitizer: odr-violation
// CHECK: Done.
