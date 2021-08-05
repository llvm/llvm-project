// RUN: %clang_hwasan -O0 -DISREAD=1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -O1 -DISREAD=1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -O2 -DISREAD=1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_hwasan -O3 -DISREAD=1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_hwasan -O0 -DISREAD=0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

// REQUIRES: stable-runtime

#include <stdlib.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char * volatile x = (char*)malloc(10);
  free(x);
  __hwasan_disable_allocator_tagging();
  fprintf(stderr, ISREAD ? "Going to do a READ\n" : "Going to do a WRITE\n");
  // CHECK: Going to do a [[TYPE:[A-Z]*]]
  int r = 0;
  if (ISREAD) r = x[5]; else x[5] = 42;  // should be on the same line.
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch on address {{.*}} at pc {{[0x]+}}[[PC:.*]]
  // CHECK: [[TYPE]] of size 1 at {{.*}} tags: [[PTR_TAG:[0-9a-f][0-9a-f]]]/[[MEM_TAG:[0-9a-f][0-9a-f]]] (ptr/mem)
  // CHECK: #{{[0-9]}} {{[0-9]+}}{{.*}}[[PC]] in main {{.*}}use-after-free.c:[[@LINE-3]]
  // Offset is 5 or 11 depending on left/right alignment.
  // CHECK: is a small unallocated heap chunk; size: 32 offset: {{5|11}}
  // CHECK: Cause: use-after-free
  // CHECK: is located 5 bytes inside of 10-byte region
  //
  // CHECK: freed by thread {{.*}} here:
  // CHECK: #0 {{.*}} in {{.*}}free{{.*}} {{.*}}hwasan_allocation_functions.cpp
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.c:[[@LINE-16]]

  // CHECK: previously allocated here:
  // CHECK: #0 {{.*}} in {{.*}}malloc{{.*}} {{.*}}hwasan_allocation_functions.cpp
  // CHECK: #1 {{.*}} in main {{.*}}use-after-free.c:[[@LINE-21]]
  // CHECK: Memory tags around the buggy address (one tag corresponds to 16 bytes):
  // CHECK: =>{{.*}}[[MEM_TAG]]
  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch
  return r;
}
