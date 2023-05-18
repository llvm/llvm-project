// Test that we can include header with TSan atomic interface.
// RUN: %clang_tsan %s -o %t && %run %t 2>&1 | FileCheck %s
#include <sanitizer/tsan_interface_atomic.h>
#include <stdio.h>

int main() {
  __tsan_atomic32 a;
  __tsan_atomic32_store(&a, 100, __tsan_memory_order_release);
  int res = __tsan_atomic32_load(&a, __tsan_memory_order_acquire);
  if (res == 100) {
    // CHECK: PASS
    fprintf(stderr, "PASS\n");
    return 0;
  }
  return 1;
}
