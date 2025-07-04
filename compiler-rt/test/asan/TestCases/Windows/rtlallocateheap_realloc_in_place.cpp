// RUN: %clang_cl_asan %Od %s %Fe%t %MD
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true:halt_on_error=false not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <windows.h>

using AllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, SIZE_T);
using ReAllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID, SIZE_T);
using FreeFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID);

int main() {
  HMODULE NtDllHandle = GetModuleHandle("ntdll.dll");
  if (!NtDllHandle) {
    puts("Couldn't load ntdll??");
    return -1;
  }

  auto RtlAllocateHeap_ptr =
      (AllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlAllocateHeap");
  if (RtlAllocateHeap_ptr == 0) {
    puts("Couldn't RtlAllocateHeap");
    return -1;
  }

  auto RtlReAllocateHeap_ptr =
      (ReAllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlReAllocateHeap");
  if (RtlReAllocateHeap_ptr == 0) {
    puts("Couldn't find RtlReAllocateHeap");
    return -1;
  }

  auto RtlFreeHeap_ptr =
      (FreeFunctionPtr)GetProcAddress(NtDllHandle, "RtlFreeHeap");
  if (RtlFreeHeap_ptr == 0) {
    puts("Couldn't RtlFreeHeap");
    return -1;
  }

  char *buffer;
  void *ret;
  buffer = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), 0, 23),

  ret = RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_REALLOC_IN_PLACE_ONLY,
                        buffer, 7);
  if (!ret) {
    puts("returned nullptr");
  }
  buffer[6] = 'a';
  puts("Okay 6");
  fflush(stdout);
  // CHECK: Okay 6

  ret = RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_REALLOC_IN_PLACE_ONLY,
                        buffer, 15);
  if (!ret) {
    puts("returned nullptr");
  }
  buffer[14] = 'a';
  puts("Okay 14");
  fflush(stdout);
  // CHECK: Okay 14

  buffer[15] = 'a';
  // CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0

  RtlFreeHeap_ptr(GetProcessHeap(), 0, buffer);
}
