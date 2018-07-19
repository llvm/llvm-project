// RUN: %clang_hwasan -O0 -DLOAD %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,LOAD
// RUN: %clang_hwasan -O1 -DLOAD %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,LOAD
// RUN: %clang_hwasan -O2 -DLOAD %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,LOAD
// RUN: %clang_hwasan -O3 -DLOAD %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,LOAD

// RUN: %clang_hwasan -O0 -DSTORE %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,STORE

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*)malloc(10);
  free(x);
  __hwasan_disable_allocator_tagging();
#ifdef STORE
  x[5] = 42;
#endif
#ifdef LOAD
  return x[5];
#endif
  // LOAD: READ of size 1 at
  // LOAD: #0 {{.*}} in main {{.*}}use-after-free.c:22

  // STORE: WRITE of size 1 at
  // STORE: #0 {{.*}} in main {{.*}}use-after-free.c:19

  // CHECK: freed here:
  // CHECK: #0 {{.*}} in {{.*}}free{{.*}} {{.*}}hwasan_interceptors.cc
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.c:16

  // CHECK: previously allocated here:
  // CHECK: #0 {{.*}} in {{.*}}malloc{{.*}} {{.*}}hwasan_interceptors.cc
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.c:15

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in main
}
