// RUN: %clang_cl_asan %Od %s %Fe%t %MD
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <windows.h>

using AllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, SIZE_T);
using ReAllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID, SIZE_T);
using FreeFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID);

int main() {
  HMODULE NtDllHandle = GetModuleHandle("ntdll.dll");
  if (!NtDllHandle) {
    fputs("Couldn't load ntdll??\n", stderr);
    return -1;
  }

  auto RtlAllocateHeap_ptr =
      (AllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlAllocateHeap");
  if (RtlAllocateHeap_ptr == 0) {
    fputs("Couldn't RtlAllocateHeap\n", stderr);
    return -1;
  }

  auto RtlReAllocateHeap_ptr =
      (ReAllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlReAllocateHeap");
  if (RtlReAllocateHeap_ptr == 0) {
    fputs("Couldn't find RtlReAllocateHeap\n", stderr);
    return -1;
  }

  auto RtlFreeHeap_ptr =
      (FreeFunctionPtr)GetProcAddress(NtDllHandle, "RtlFreeHeap");
  if (RtlFreeHeap_ptr == 0) {
    fputs("Couldn't RtlFreeHeap\n", stderr);
    return -1;
  }

  char *ptr1;
  char *ptr2;
  ptr2 = ptr1 = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), 0, 15);
  if (ptr1)
    fputs("Okay alloc\n", stderr);
  // CHECK: Okay alloc

  // TODO: Growing is currently not supported
  ptr2 = (char *)RtlReAllocateHeap_ptr(GetProcessHeap(),
                                       HEAP_REALLOC_IN_PLACE_ONLY, ptr1, 23);
  if (ptr2 == NULL)
    fputs("Okay grow failed\n", stderr);
  // CHECK: Okay grow failed

  // TODO: Shrinking is currently not supported
  ptr2 = (char *)RtlReAllocateHeap_ptr(GetProcessHeap(),
                                       HEAP_REALLOC_IN_PLACE_ONLY, ptr1, 7);
  if (ptr2 == ptr1)
    fputs("Okay shrinking return the original pointer\n", stderr);
  // CHECK: Okay shrinking return the original pointer

  ptr1[7] = 'a';
  fputs("Okay 7\n", stderr);
  // CHECK: Okay 7

  // TODO: Writing behind the shrinked part is currently not detected.
  //       Therefore test writing behind the original allocation for now.
  ptr1[16] = 'a';
  // CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0

  RtlFreeHeap_ptr(GetProcessHeap(), 0, ptr1);
}
