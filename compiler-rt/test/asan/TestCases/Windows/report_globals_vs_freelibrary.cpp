// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -LD -Od -DDLL %s -Fe%t.dll
// RUN: %clang_cl_asan -Od -DEXE %s -Fe%te.exe
// RUN: %env_asan_opts=report_globals=2 %run %te.exe %t.dll 2>&1 | FileCheck %s

#include <windows.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#if defined(EXE)
int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s [client].dll\n", argv[0]);
    return 101;
  }
  const char *dll_name = argv[1];

// CHECK: time to load DLL
  printf("time to load DLL\n");
  fflush(0);

// On DLL load, the "in DLL\n" string is registered:
// CHECK: Added Global{{.*}} size=19
// CHECK: in DLL(reason=1)
  HMODULE dll = LoadLibrary(dll_name);
  if (dll == NULL)
    return 3;

// CHECK: in DLL(reason=0)
// CHECK-NEXT: Removed Global{{.*}} size=19
  if (!FreeLibrary(dll))
    return 4;

// CHECK: bye!
  printf("bye!\n");
  fflush(0);
}
#elif defined(DLL)
BOOL WINAPI DllMain(HMODULE, DWORD reason, LPVOID) {
  printf("in DLL(reason=%d)\n", (int)reason);
  fflush(0);
  return TRUE;
}
#else
# error oops!
#endif
}
