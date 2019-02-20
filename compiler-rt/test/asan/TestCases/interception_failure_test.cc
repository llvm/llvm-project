// If user provides his own libc functions, ASan doesn't
// intercept these functions.

// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s
// XFAIL: freebsd

// On Windows, defining strtoll in a static build results in linker errors, but
// it works with the dynamic runtime.
// XFAIL: win32-static-asan

// On NetBSD, defining strtol in a static build results in linker errors, but
// it works with the dynamic runtime.
// XFAIL: netbsd && !asan-dynamic-runtime

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" long strtol(const char *nptr, char **endptr, int base) {
  fprintf(stderr, "my_strtol_interceptor\n");
  if (endptr)
    *endptr = (char*)nptr + strlen(nptr);
  return 0;
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return (int)strtol(x, 0, 10);
  // CHECK: my_strtol_interceptor
  // CHECK-NOT: heap-use-after-free
}
