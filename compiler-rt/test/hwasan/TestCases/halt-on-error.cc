// RUN: %clangxx_hwasan -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  int* volatile x = (int*)malloc(16);
  free(x);
  __hwasan_disable_allocator_tagging();
  return x[2] + ((char *)x)[6] + ((char *)x)[9];
  // CHECK: READ of size 4 at
  // CHECK: #0 {{.*}} in main {{.*}}halt-on-error.cc:12
  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main

  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main {{.*}}halt-on-error.cc:12
  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main

  // CHECK: READ of size 1 at
  // CHECK: #0 {{.*}} in main {{.*}}halt-on-error.cc:12
  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main

  // CHECK-NOT: tag-mismatch
}
