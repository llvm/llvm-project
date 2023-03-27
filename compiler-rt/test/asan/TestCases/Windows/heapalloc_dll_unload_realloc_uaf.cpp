// UNSUPPORTED: target={{.*-windows-gnu}}

#include <stdio.h>
#include <windows.h>

// RUN: %clang_cl_asan -LD /Od -DDLL %s -Fe%t.dll
// RUN: %clang_cl /Od -DEXE %s -Fe%te.exe
// RUN: %env_asan_opts=windows_hook_rtl_allocators=true not %run %te.exe %t.dll 2>&1 | FileCheck %s
// REQUIRES: asan-dynamic-runtime
// REQUIRES: asan-32-bits

#include <cassert>
#include <stdio.h>
#include <windows.h>
extern "C" {
#if defined(EXE)

int main(int argc, char **argv) {
  void *region_without_hooks = HeapAlloc(GetProcessHeap(), 0, 10);
  HMODULE lib = LoadLibraryA(argv[1]);
  assert(lib != INVALID_HANDLE_VALUE);
  assert(0 != FreeLibrary(lib));
  assert(0 != HeapFree(GetProcessHeap(), 0, region_without_hooks));
  HeapReAlloc(GetProcessHeap(), 0, region_without_hooks, 100); //should throw nested error
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
