// Regression test for PR33206
//
// RUN: %clang -DDYN=1 -DMALLOC=1 -fPIC -shared %s -o %t-dso1.so
// RUN: %clang -DDYN=1 -DMALLOC=1 -fPIC -shared %s -o %t-dso2.so %t-dso1.so
// RUN: %clang %s -o %t-1 %t-dso2.so
// RUN: env LD_PRELOAD=%shared_libasan %run %t-1 2>&1 | FileCheck %s
// RUN: %clang -DDYN=1 -DREALLOC=1 -fPIC -shared %s -o %t-dso3.so
// RUN: %clang -DDYN=1 -DREALLOC=1 -fPIC -shared %s -o %t-dso4.so %t-dso3.so
// RUN: %clang %s -o %t-2 %t-dso4.so
// RUN: env LD_PRELOAD=%shared_libasan %run %t-2 2>&1 | FileCheck %s
// REQUIRES: asan-dynamic-runtime

// FIXME: Test regressed while android bot was disabled. Needs investigation.
// UNSUPPORTED: android

#include <stdlib.h>
#include <stdio.h>

#ifdef DYN
__attribute__((constructor)) void foo() {
  void *p;
#ifdef MALLOC
  p = malloc(1 << 20);
#endif
#ifdef REALLOC
  p = realloc (0, 1 << 20);
#endif
  free(p);
}
#else
int main() {
  // CHECK: Success
  printf("Success\n");
  return 0;
}
#endif
