// UNSUPPORTED: target={{.*-windows-gnu}}

#include "sanitizer\allocator_interface.h"
#include <cassert>
#include <stdio.h>
#include <windows.h>
// RUN: %clang_cl_asan %s -o%t
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits

int main() {
  //owned by rtl
  void *alloc = HeapAlloc(GetProcessHeap(), HEAP_GENERATE_EXCEPTIONS, 100);
  assert(alloc);
  // still owned by rtl
  alloc = HeapReAlloc(GetProcessHeap(), HEAP_GENERATE_EXCEPTIONS, alloc, 100);
  assert(alloc && !__sanitizer_get_ownership(alloc) && HeapValidate(GetProcessHeap(), 0, alloc));
  //convert to asan owned
  void *realloc = HeapReAlloc(GetProcessHeap(), 0, alloc, 500);
  alloc = nullptr;
  assert(realloc && __sanitizer_get_ownership(realloc));
  //convert back to rtl owned;
  alloc = HeapReAlloc(GetProcessHeap(), HEAP_GENERATE_EXCEPTIONS, realloc, 100);
  assert(alloc && !__sanitizer_get_ownership(alloc) && HeapValidate(GetProcessHeap(), 0, alloc));
  printf("Success\n");
}

// CHECK-NOT: assert
// CHECK-NOT: AddressSanitizer
// CHECK: Success