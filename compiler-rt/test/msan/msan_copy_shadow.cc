// Test that __msan_copy_shadow copies shadow, updates origin and does not touch
// the application memory.
// RUN: %clangxx_msan -fsanitize-memory-track-origins=0 -O0 %s -o %t && not %run %t 2>&1
// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O0 %s -o %t && not %run %t 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-%short-stack %s

#include <assert.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

int main() {
  char *a = new char[4];
  char *b = new char[4];
  a[1] = 1;
  a[3] = 2;
  memset(b, 42, 4);

  // Test that __msan_copy_shadow does not touch the contents of b[].
  __msan_copy_shadow(b, a, 4);
  __msan_unpoison(b, 4);
  assert(b[0] == 42 && b[1] == 42 && b[2] == 42 && b[3] == 42);

  // Test that __msan_copy_shadow correctly updates shadow and origin of b[].
  __msan_copy_shadow(b, a, 4);
  assert(__msan_test_shadow(b, 4) == 0);
  assert(__msan_test_shadow(b + 1, 3) == 1);
  assert(__msan_test_shadow(b + 3, 1) == -1);
  __msan_check_mem_is_initialized(b, 4);
  // CHECK: use-of-uninitialized-value
  // CHECK:   {{in main.*msan_copy_shadow.cc:}}[[@LINE-2]]
  // CHECK: Uninitialized value was stored to memory at
  // CHECK-FULL-STACK:   {{in main.*msan_copy_shadow.cc:}}[[@LINE-8]]
  // CHECK-SHORT-STACK:   {{in __msan_copy_shadow .*msan_interceptors.cpp:}}
  // CHECK: Uninitialized value was created by a heap allocation
  // CHECK:   {{in main.*msan_copy_shadow.cc:}}[[@LINE-23]]
}
