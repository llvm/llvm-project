// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void
__asan_on_error() {
  fprintf(stderr, "__asan_on_error called\n");
  fflush(stderr);
}

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: __asan_on_error called
}
