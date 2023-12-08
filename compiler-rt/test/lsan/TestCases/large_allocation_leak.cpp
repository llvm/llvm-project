// Test that LargeMmapAllocator's chunks aren't reachable via some internal data structure.
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=report_objects=1:use_stacks=0:use_registers=0 not %run %t 2>&1 | FileCheck %s

// For 32 bit LSan it's pretty likely that large chunks are "reachable" from some
// internal data structures (e.g. Glibc global data).
// UNSUPPORTED: i686,target={{(x86|arm).*}}

#include <stdio.h>
#include <stdlib.h>
#include "sanitizer_common/print_address.h"

int main() {
  // maxsize in primary allocator is always less than this (1 << 25).
  void *large_alloc = malloc(33554432);
  print_address("Test alloc: ", 1, large_alloc);
  return 0;
}
// CHECK: Test alloc: [[ADDR:0x[0-9,a-f]+]]
// CHECK: LeakSanitizer: detected memory leaks
// CHECK: [[ADDR]] (33554432 bytes)
// CHECK: SUMMARY: {{.*}}Sanitizer:
