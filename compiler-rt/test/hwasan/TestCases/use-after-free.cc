// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_hwasan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_hwasan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_hwasan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char *x = (char*)malloc(10);
  free(x);
  __hwasan_disable_allocator_tagging();
  return x[5];
  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main {{.*}}use-after-free.cc:15

  // CHECK: freed here:
  // CHECK: #0 {{.*}} in free {{.*}}hwasan_interceptors.cc
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.cc:13

  // CHECK: previously allocated here:
  // CHECK: #0 {{.*}} in __interceptor_malloc {{.*}}hwasan_interceptors.cc
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.cc:12

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
