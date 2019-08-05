// RUN: %clang_cl_asan -Od %s -Fe%t /MD
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: asan-64-bits
// REQUIRES: asan-rtl-heap-interception

#include <assert.h>
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

  auto RtlAllocateHeap_ptr = (AllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlAllocateHeap");
  if (RtlAllocateHeap_ptr == 0) {
    puts("Couldn't find RtlAllocateHeap");
    return -1;
  }

  auto RtlReAllocateHeap_ptr = (ReAllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlReAllocateHeap");
  if (RtlReAllocateHeap_ptr == 0) {
    puts("Couldn't find RtlReAllocateHeap");
    return -1;
  }

  char *buffer;
  SIZE_T buffer_size = 32;
  SIZE_T new_buffer_size = buffer_size * 2;

  buffer = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), HEAP_ZERO_MEMORY, buffer_size);
  assert(buffer != nullptr);
  // Check that the buffer is zeroed.
  for (SIZE_T i = 0; i < buffer_size; ++i) {
    assert(buffer[i] == 0);
  }
  memset(buffer, 0xcc, buffer_size);

  // Zero the newly allocated memory.
  buffer = (char *)RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_ZERO_MEMORY, buffer, new_buffer_size);
  assert(buffer != nullptr);
  // Check that the first part of the buffer still has the old contents.
  for (SIZE_T i = 0; i < buffer_size; ++i) {
    assert(buffer[i] == (char)0xcc);
  }
  // Check that the new part of the buffer is zeroed.
  for (SIZE_T i = buffer_size; i < new_buffer_size; ++i) {
    assert(buffer[i] == 0x0);
  }

  // Shrink the buffer back down.
  buffer = (char *)RtlReAllocateHeap_ptr(GetProcessHeap(), HEAP_ZERO_MEMORY, buffer, buffer_size);
  assert(buffer != nullptr);
  // Check that the first part of the buffer still has the old contents.
  for (SIZE_T i = 0; i < buffer_size; ++i) {
    assert(buffer[i] == (char)0xcc);
  }

  buffer[buffer_size + 1] = 'a';
  // CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0
}
