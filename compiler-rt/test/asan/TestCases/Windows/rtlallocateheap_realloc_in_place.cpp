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
  buffer = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), 0, 48),

  RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_REALLOC_IN_PLACE_ONLY, buffer,
                        16);
  buffer[15] = 'a';
  puts("Okay 15");
  fflush(stdout);
  // CHECK: Okay 15

  RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_REALLOC_IN_PLACE_ONLY, buffer,
                        32);
  buffer[31] = 'a';
  puts("Okay 31");
  fflush(stdout);
  // CHECK: Okay 31

  buffer[32] = 'a';
  // CHECK: AddressSanitizer: use-after-poison on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0

  RtlFreeHeap_ptr(GetProcessHeap(), 0, buffer);
}
