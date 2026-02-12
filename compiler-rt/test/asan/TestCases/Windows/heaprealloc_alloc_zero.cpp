// RUN: %clang_cl_asan %Od %MT -o %t %s
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
#include <cassert>
#include <iostream>
#include <sanitizer/allocator_interface.h>
#include <windows.h>

int main() {
  void *ptr = malloc(0);
  if (ptr)
    std::cerr << "allocated!\n";

  // Zero-size allocations are internally upgraded to size 1.
  // __sanitizer_get_allocated_size reports bytes reserved, not requested.
  // Dereferencing the pointer will be detected as a heap-buffer-overflow.
  if (__sanitizer_get_allocated_size(ptr) != 1)
    return 1;

  free(ptr);

  // Zero-size HeapAlloc/HeapReAlloc should report size 0 via HeapSize,
  // matching the originally requested size.
  ptr = HeapAlloc(GetProcessHeap(), 0, 0);
  if (!ptr)
    return 1;
  void *ptr2 = HeapReAlloc(GetProcessHeap(), 0, ptr, 0);
  if (!ptr2)
    return 1;
  size_t heapsize = HeapSize(GetProcessHeap(), 0, ptr2);
  if (heapsize != 0) {
    std::cerr << "HeapAlloc size failure! " << heapsize << " != 0\n";
    return 1;
  }
  void *ptr3 = HeapReAlloc(GetProcessHeap(), 0, ptr2, 3);
  if (!ptr3)
    return 1;
  heapsize = HeapSize(GetProcessHeap(), 0, ptr3);

  if (heapsize != 3) {
    std::cerr << "HeapAlloc size failure! " << heapsize << " != 3\n";
    return 1;
  }
  HeapFree(GetProcessHeap(), 0, ptr3);
  return 0;
}

// CHECK: allocated!
// CHECK-NOT: heap-buffer-overflow
// CHECK-NOT: AddressSanitizer
// CHECK-NOT: HeapAlloc size failure!
