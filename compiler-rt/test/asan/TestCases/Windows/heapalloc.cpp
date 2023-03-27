// UNSUPPORTED: target={{.*-windows-gnu}}

// UNSUPPORTED: asan-64-bits
// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s

#include <windows.h>

int main() {
  char *buffer;
  buffer = (char *)HeapAlloc(GetProcessHeap(), 0, 32),
  buffer[33] = 'a';
  // CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0
}
