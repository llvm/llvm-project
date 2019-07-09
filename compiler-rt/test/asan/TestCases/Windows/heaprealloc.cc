// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
#include <stdio.h>
#include <windows.h>

int main() {
  char *oldbuf;
  size_t sz = 8;
  HANDLE procHeap = GetProcessHeap();
  oldbuf = (char *)HeapAlloc(procHeap, 0, sz);
  char *newbuf = oldbuf;
  while (oldbuf == newbuf) {
    sz *= 2;
    newbuf = (char *)HeapReAlloc(procHeap, 0, oldbuf, sz);
  }

  newbuf[0] = 'a';
  oldbuf[0] = 'a';
  // CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[WRITE2:0x[0-9a-f]+]] thread T0
  // CHECK: #0 {{0x[0-9a-f]+ in main.*}}:[[@LINE-3]]
}
