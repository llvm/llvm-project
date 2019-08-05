// RUN: %clang_cl_asan /Od -o %t %s
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=windows_hook_rtl_allocators=false %run %t 2>&1 | FileCheck %s
// RUN: %clang_cl /Od -o %t %s
// RUN: %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
#include <cassert>
#include <stdio.h>
#include<windows.h>

int main() {
  HANDLE heap = HeapCreate(0, 0, 0);
  void *ptr = HeapAlloc(heap, 0, 4);
  assert(ptr);
  void *ptr2 = HeapReAlloc(heap, 0, ptr, 0);
  assert(ptr2);
  HeapFree(heap, 0, ptr2);
  fprintf(stderr, "passed!\n");
}

// CHECK-NOT: double-free
// CHECK-NOT: AddressSanitizer
// CHECK: passed!