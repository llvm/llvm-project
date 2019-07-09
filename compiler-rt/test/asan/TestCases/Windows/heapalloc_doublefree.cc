// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
#include <cassert>
#include <windows.h>

int main() {
  void *allocation = HeapAlloc(GetProcessHeap(), 0, 10);
  assert(allocation != 0);
  assert(HeapFree(GetProcessHeap(), 0, allocation));
  HeapFree(GetProcessHeap(), 0, allocation); //will dump
  assert(0 && "HeapFree double free should produce an ASAN dump\n");
  return 0;
}

// CHECK: AddressSanitizer: attempting double-free on [[addr:0x[0-9a-fA-F]+]] in thread T0: