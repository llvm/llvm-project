// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: %env_asan_opts=allocator_may_return_null=true %run %t
// RUN: %env_asan_opts=allocator_may_return_null=true:windows_hook_rtl_allocators=true %run %t
// UNSUPPORTED: asan-64-bits
#include <windows.h>
int main() {
  void *nope = HeapAlloc(GetProcessHeap(), 0, ((size_t)0) - 1);
  return (nope == nullptr) ? 0 : 1;
}