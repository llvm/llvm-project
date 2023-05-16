// RUN: %clang_cl_asan %LD %Od -DDLL %s %Fe%t.dll
// RUN: %clang_cl %Od -DEXE %s %Fe%te.exe
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %te.exe %t.dll 2>&1 | FileCheck %s
// REQUIRES: asan-dynamic-runtime
// REQUIRES: asan-32-bits

#include <cassert>
#include <stdio.h>
#include <windows.h>

extern "C" {
#if defined(EXE)
using AllocateFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, SIZE_T);
using FreeFunctionPtr = PVOID(__stdcall *)(PVOID, ULONG, PVOID);

int main(int argc, char **argv) {
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

  auto RtlFreeHeap_ptr =
      (FreeFunctionPtr)GetProcAddress(NtDllHandle, "RtlFreeHeap");
  if (RtlFreeHeap_ptr == 0) {
    puts("Couldn't get RtlFreeHeap");
    return -1;
  }

  char *buffer;
  buffer = (char *)RtlAllocateHeap_ptr(GetProcessHeap(), 0, 32);

  HMODULE lib = LoadLibraryA(argv[1]);
  assert(lib != INVALID_HANDLE_VALUE);
  assert(0 != FreeLibrary(lib));

  if (!RtlFreeHeap_ptr(GetProcessHeap(), 0, buffer)) {
    puts("Couldn't RtlFreeHeap");
    return -1;
  }
  // Because this pointer was allocated pre-hooking,
  // this will dump as a nested bug. Asan attempts to free
  // the pointer and AV's, so the ASAN exception handler
  // will dump as a 'nested bug'.
  RtlFreeHeap_ptr(GetProcessHeap(), 0, buffer);
}

#elif defined(DLL)
// This global is registered at startup.

BOOL WINAPI DllMain(HMODULE, DWORD reason, LPVOID) {
  fprintf(stderr, "in DLL(reason=%d)\n", (int)reason);
  fflush(0);
  return TRUE;
}

// CHECK: in DLL(reason=1)
// CHECK: in DLL(reason=0)
// CHECK: AddressSanitizer: nested bug in the same thread, aborting.

#else
#error oops!
#endif
}
