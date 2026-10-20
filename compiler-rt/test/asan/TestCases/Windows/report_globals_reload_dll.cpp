// Make sure we can handle reloading the same DLL multiple times.
// RUN: %clang_cl_asan %LD %Od -DDLL %s %Fe%t.dll
// RUN: %clang_cl_asan %Od -DEXE %s %Fe%te.exe
// RUN: %env_asan_opts=report_globals=1 %run %te.exe %t.dll 2>&1 | FileCheck %s

#include <windows.h>
#include <stdio.h>
#include <string.h>

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

// CHECK: in DLL(reason=1)
// CHECK: in DLL(reason=0)
// CHECK: in DLL(reason=1)
// CHECK: in DLL(reason=0)
// CHECK: in DLL(reason=1)
// CHECK: in DLL(reason=0)
  for (int i = 0; i < 30; ++i) {
    HMODULE dll = LoadLibrary(dll_name);
    if (dll == NULL)
      return 3;

    if (!FreeLibrary(dll))
      return 4;
  }

// CHECK: All OK!
  printf("All OK!\n");
  fflush(0);
}
#elif defined(DLL)
extern "C" {
BOOL WINAPI DllMain(HMODULE, DWORD reason, LPVOID) {
  printf("in DLL(reason=%d)\n", (int)reason);
  fflush(0);
  return TRUE;
}
}
#else
# error oops!
#endif
