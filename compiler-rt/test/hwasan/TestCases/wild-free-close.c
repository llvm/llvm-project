// RUN: %clang_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();
  char *p = (char *)malloc(1);
  fprintf(stderr, "ALLOC %p\n", __hwasan_tag_pointer(p, 0));
  // CHECK: ALLOC {{[0x]+}}[[ADDR:.*]]
  free(p - 8);
  // CHECK: ERROR: HWAddressSanitizer: invalid-free on address {{.*}} at pc {{[0x]+}}[[PC:.*]] on thread T{{[0-9]+}}
  // CHECK: #0 {{[0x]+}}{{.*}}[[PC]] in free
  // CHECK: #1 {{.*}} in main {{.*}}wild-free-close.c:[[@LINE-3]]
  // CHECK: is located 8 bytes to the left of 1-byte region [{{[0x]+}}{{.*}}[[ADDR]]
  // CHECK-NOT: Segmentation fault
  // CHECK-NOT: SIGSEGV
  return 0;
}
