// RUN: %clang_cl_asan -O0 %s -Fe%t /MD
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
// REQUIRES: asan-rtl-heap-interception

#include <assert.h>
#include <stdio.h>
#include <windows.h>

extern "C" int __sanitizer_get_ownership(const volatile void *p);
using AllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, SIZE_T);
using FreeFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID);

int main() {
  HMODULE NtDllHandle = GetModuleHandle("ntdll.dll");
  if (!NtDllHandle) {
    puts("Couldn't load ntdll??");
    return -1;
  }

  auto RtlAllocateHeap_ptr = (AllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlAllocateHeap");
  if (RtlAllocateHeap_ptr == 0) {
    puts("Couldn't RtlAllocateHeap");
    return -1;
  }

  auto RtlFreeHeap_ptr = (FreeFunctionPtr)GetProcAddress(NtDllHandle, "RtlFreeHeap");
  if (RtlFreeHeap_ptr == 0) {
    puts("Couldn't RtlFreeHeap");
    return -1;
  }

  char *winbuf;
  char *asanbuf;
  winbuf = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), HEAP_GENERATE_EXCEPTIONS, 32),
  asanbuf = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), 0, 32),
  winbuf[0] = 'a';
  assert(!__sanitizer_get_ownership(winbuf));
  assert(__sanitizer_get_ownership(asanbuf));

  RtlFreeHeap_ptr(GetProcessHeap(), 0, winbuf);
  RtlFreeHeap_ptr(GetProcessHeap(), 0, asanbuf);
  puts("Okay");
  // CHECK: Okay
}
