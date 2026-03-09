// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-ESCAPE
// RUN: %clangxx_lowfat -O0 %s -DSENTINEL -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-SENTINEL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Use noinline to prevent the compiler from optimizing away the GEP.
__attribute__((noinline)) void sink(void *p) {
  printf("p = %p\n", p);
}

int main() {
  char *p = (char *)malloc(16);
  if (!p) return 1;

#ifdef SENTINEL
  // Test the "base - 1" sentinel idiom.
  // CHECK-SENTINEL: LOWFAT ERROR: out-of-bounds error detected!
  char *sentinel = p - 1;
  sink(sentinel);
#else
  // Test pointer escaping an allocation boundary.
  // CHECK-ESCAPE: LOWFAT ERROR: out-of-bounds error detected!
  char *escaped = p + 16;
  sink(escaped);
#endif

  free(p);
  return 0;
}
