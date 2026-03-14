// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
#include <assert.h>
#include <stdio.h>
#include <windows.h>

extern "C" int
__sanitizer_get_ownership(const volatile void *p);

int main() {
  char *buffer;
  buffer = (char *)HeapAlloc(GetProcessHeap(), HEAP_GENERATE_EXCEPTIONS, 32);
  buffer[0] = 'a';
  assert(!__sanitizer_get_ownership(buffer));
  HeapFree(GetProcessHeap(), 0, buffer);
  puts("Okay");
  // CHECK: Okay
}
