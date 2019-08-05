// RUN: %clangxx_hwasan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=COMMON
// RUN: %clangxx_hwasan -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=1 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clangxx_hwasan -O0 %s -o %t -fsanitize-recover=hwaddress && not %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -O0 %s -o %t -fsanitize-recover=hwaddress && not %env_hwasan_opts=halt_on_error=1 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -O0 %s -o %t -fsanitize-recover=hwaddress && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefixes=COMMON,RECOVER

// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=1 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON

// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t -fsanitize-recover=hwaddress && not %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t -fsanitize-recover=hwaddress && not %env_hwasan_opts=halt_on_error=1 %run %t 2>&1 | FileCheck %s --check-prefix=COMMON
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t -fsanitize-recover=hwaddress && not %env_hwasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s --check-prefixes=COMMON,RECOVER

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  int* volatile x = (int*)malloc(16);
  free(x);
  __hwasan_disable_allocator_tagging();
  return x[2] + ((char *)x)[6] + ((char *)x)[9];
  // COMMON: READ of size 4 at
  // When instrumenting with callbacks, main is actually #1, and #0 is __hwasan_load4.
  // COMMON: #{{.*}} in main {{.*}}halt-on-error.cpp:27
  // COMMON: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in

  // RECOVER: READ of size 1 at
  // RECOVER: #{{.*}} in main {{.*}}halt-on-error.cpp:27
  // RECOVER: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in

  // RECOVER: READ of size 1 at
  // RECOVER: #{{.*}} in main {{.*}}halt-on-error.cpp:27
  // RECOVER: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in

  // COMMON-NOT: tag-mismatch
}
