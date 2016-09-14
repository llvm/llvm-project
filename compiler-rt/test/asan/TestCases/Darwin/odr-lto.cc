// Check that -asan-use-private-alias and use_odr_indicator=1 silence the false
// positive ODR violation on Darwin with LTO.

// REQUIRES: lto

// RUN: %clangxx_asan -DPART=0 -c %s -o %t-1.o -flto
// RUN: %clangxx_asan -DPART=1 -c %s -o %t-2.o -flto
// RUN: %clangxx_asan %t-1.o %t-2.o -o %t -flto
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ODR

// RUN: %clangxx_asan -DPART=0 -c %s -o %t-1.o -flto -mllvm -asan-use-private-alias
// RUN: %clangxx_asan -DPART=1 -c %s -o %t-2.o -flto -mllvm -asan-use-private-alias
// RUN: %clangxx_asan %t-1.o %t-2.o -o %t -flto
// RUN: %env_asan_opts=use_odr_indicator=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ODR

#include <stdio.h>
#include <stdlib.h>
void putstest();

#if PART == 1

int main()
{
  fputs("test\n", stderr);
  putstest();
  fprintf(stderr, "Done.\n");
  return 0;
}

#else // PART == 1

void putstest()
{
  fputs("test\n", stderr);
}

#endif // PART == 1

// CHECK-ODR: ERROR: AddressSanitizer: odr-violation
// CHECK-NO-ODR-NOT: ERROR: AddressSanitizer: odr-violation
// CHECK-NO-ODR: Done.
