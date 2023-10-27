// RUN: %clang_cl_asan %Od %MT -o %t %s
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
#include <cassert>
#include <iostream>
#include <windows.h>

int main() {
  void *ptr = malloc(0);
  if (ptr)
    std::cerr << "allocated!\n";
  ((char *)ptr)[0] = '\xff'; //check this 'allocate 1 instead of 0' hack hasn't changed

  free(ptr);

  /*
        HeapAlloc hack for our asan interceptor is to change 0
        sized allocations to size 1 to avoid weird inconsistencies
        between how realloc and heaprealloc handle 0 size allocations.

        Note this test relies on these instructions being intercepted.
        Without ASAN HeapRealloc on line 27 would return a ptr whose
        HeapSize would be 0. This test makes sure that the underlying behavior
        of our hack hasn't changed underneath us.

        We can get rid of the test (or change it to test for the correct
        behavior) once we fix the interceptor or write a different allocator
        to handle 0 sized allocations properly by default.

    */
  ptr = HeapAlloc(GetProcessHeap(), 0, 0);
  if (!ptr)
    return 1;
  void *ptr2 = HeapReAlloc(GetProcessHeap(), 0, ptr, 0);
  if (!ptr2)
    return 1;
  size_t heapsize = HeapSize(GetProcessHeap(), 0, ptr2);
  if (heapsize != 1) { // will be 0 without ASAN turned on
    std::cerr << "HeapAlloc size failure! " << heapsize << " != 1\n";
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
