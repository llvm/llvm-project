// Build a library with origin tracking and an executable w/o origin tracking.
// Test that origin tracking is enabled at runtime.
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_msan -O0 %s %t-so.so -o %t && not %run %t 2>&1 | FileCheck %s

#ifdef BUILD_SO

#include <stdlib.h>

extern "C" {
void my_access(int *p) {
  volatile int tmp;
  // Force initialize-ness check.
  if (*p)
    tmp = 1;
}

void *my_alloc(unsigned sz) {
  return malloc(sz);
}
}  // extern "C"

#else  // BUILD_SO

#include <stdlib.h>

extern "C" {
void my_access(int *p);
void *my_alloc(unsigned sz);
}

int main(int argc, char **argv) {
  int *x = (int *)my_alloc(sizeof(int));
  my_access(x);
  delete x;

  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in my_access .*dso-origin.cpp:}}
  // CHECK: {{#1 0x.* in main .*dso-origin.cpp:}}[[@LINE-5]]
  // CHECK: Uninitialized value was created by a heap allocation
  // CHECK: {{#0 0x.* in .*malloc}}
  // CHECK: {{#1 0x.* in my_alloc .*dso-origin.cpp:}}
  // CHECK: {{#2 0x.* in main .*dso-origin.cpp:}}[[@LINE-10]]
  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*dso-origin.cpp:.* my_access}}
  return 0;
}

#endif  // BUILD_SO
