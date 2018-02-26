// Test that non-sanitized executables work with sanitized shared libs
// and preloaded runtime.
//
// RUN: %clangxx -DBUILD_SO=1 -fPIC -shared %s -o %t.so
// RUN: %clangxx %s %t.so -o %t
//
// RUN: %clangxx_asan -DBUILD_SO=1 -fPIC -shared %s -o %t.so
// RUN: env LD_PRELOAD=%shared_libasan not %run %t 2>&1 | FileCheck %s

// REQUIRES: asan-dynamic-runtime

// This way of setting LD_PRELOAD does not work with Android test runner.
// REQUIRES: !android

#if BUILD_SO
char dummy;
void do_access(const void *p) {
  // CHECK: AddressSanitizer: heap-buffer-overflow
  dummy = ((const char *)p)[1];
}
#else
#include <stdlib.h>
extern void do_access(const void *p);
int main(int argc, char **argv) {
  void *p = malloc(1);
  do_access(p);
  free(p);
  return 0;
}
#endif
