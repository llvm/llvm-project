// RUN: %clang_cl_asan %s %Fe%t
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true %run %t 2>&1 | FileCheck %s
// RUN: %clang_cl_asan %s %Fe%t -DFAIL_CHECK
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
//
// Verify that zero-size heap allocations report size 0 through Windows heap
// size APIs, preventing false positives when the result is used with memset.

#include <malloc.h>
#include <stdio.h>
#include <windows.h>

using AllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, SIZE_T);
using FreeFunctionPtr = BOOL(__stdcall *)(PVOID, ULONG, PVOID);
using SizeFunctionPtr = SIZE_T(__stdcall *)(PVOID, ULONG, PVOID);

int main() {
  HMODULE NtDllHandle = GetModuleHandle("ntdll.dll");
  if (!NtDllHandle) {
    puts("Couldn't load ntdll");
    return -1;
  }

  auto RtlAllocateHeap_ptr =
      (AllocateFunctionPtr)GetProcAddress(NtDllHandle, "RtlAllocateHeap");
  auto RtlFreeHeap_ptr =
      (FreeFunctionPtr)GetProcAddress(NtDllHandle, "RtlFreeHeap");
  auto RtlSizeHeap_ptr =
      (SizeFunctionPtr)GetProcAddress(NtDllHandle, "RtlSizeHeap");

  if (!RtlAllocateHeap_ptr || !RtlFreeHeap_ptr || !RtlSizeHeap_ptr) {
    puts("Couldn't find Rtl heap functions");
    return -1;
  }

  // Test RtlAllocateHeap with zero size
  {
    char *buffer =
        (char *)RtlAllocateHeap_ptr(GetProcessHeap(), HEAP_ZERO_MEMORY, 0);
    if (buffer) {
      auto size = RtlSizeHeap_ptr(GetProcessHeap(), 0, buffer);
      memset(buffer, 0, size);
#ifdef FAIL_CHECK
      // heap-buffer-overflow since actual size is 0
      memset(buffer, 0, 1);
#endif
      RtlFreeHeap_ptr(GetProcessHeap(), 0, buffer);
    }
  }

  // Test malloc with zero size
  {
    char *buffer = (char *)malloc(0);
    if (buffer) {
      auto size = _msize(buffer);
      auto rtl_size = RtlSizeHeap_ptr(GetProcessHeap(), 0, buffer);
      memset(buffer, 0, size);
      memset(buffer, 0, rtl_size);
      free(buffer);
    }
  }

  // Test operator new with zero size
  {
    char *buffer = new char[0];
    auto size = _msize(buffer);
    auto rtl_size = RtlSizeHeap_ptr(GetProcessHeap(), 0, buffer);
    memset(buffer, 0, size);
    memset(buffer, 0, rtl_size);
    delete[] buffer;
  }

  // Test GlobalAlloc with zero size.
  // GlobalAlloc calls RtlAllocateHeap internally.
  {
    HGLOBAL hMem = GlobalAlloc(GMEM_FIXED | GMEM_ZEROINIT, 0);
    if (hMem) {
      char *buffer = (char *)hMem;
      auto size = GlobalSize(hMem);
      auto rtl_size = RtlSizeHeap_ptr(GetProcessHeap(), 0, buffer);
      memset(buffer, 0, size);
      memset(buffer, 0, rtl_size);
      GlobalFree(hMem);
    }
  }

  // Test LocalAlloc with zero size.
  // LocalAlloc calls RtlAllocateHeap internally.
  {
    HLOCAL hMem = LocalAlloc(LMEM_FIXED | LMEM_ZEROINIT, 0);
    if (hMem) {
      char *buffer = (char *)hMem;
      auto size = LocalSize(hMem);
      auto rtl_size = RtlSizeHeap_ptr(GetProcessHeap(), 0, buffer);
      memset(buffer, 0, size);
      memset(buffer, 0, rtl_size);
      LocalFree(hMem);
    }
  }

  puts("Success");
  return 0;
}

// CHECK: Success
// CHECK-NOT: AddressSanitizer: heap-buffer-overflow
// CHECK-FAIL: AddressSanitizer: heap-buffer-overflow
