// RUN: %clang_cl_asan %LD %Od -DDLL %s %Fe%t.dll \
// RUN:   %if target={{.*-windows-gnu}} %{ -Wl,--out-implib,%t.lib %}
// RUN: %clang_cl_asan %Od -DEXE %s %t.lib %Fe%te.exe
// RUN: %env_asan_opts=report_globals=2 %run %te.exe 2>&1 | FileCheck %s

// FIXME: Currently, the MT runtime build crashes on startup due to dbghelp.dll
// initialization failure.
// REQUIRES: asan-dynamic-runtime

#include <windows.h>
#include <stdio.h>

#if defined(EXE)
extern "C" __declspec(dllimport) int foo_from_dll();

// CHECK: in DLL(reason=1)
int main(int argc, char **argv) {
  foo_from_dll();
// CHECK: hello!
  printf("hello!\n");
  fflush(0);
// CHECK: in DLL(reason=0)
}
#elif defined(DLL)
extern "C" {
// This global is registered at startup.
int x[42];

__declspec(dllexport) int foo_from_dll() {
  return x[2];
}

BOOL WINAPI DllMain(HMODULE, DWORD reason, LPVOID) {
  printf("in DLL(reason=%d)\n", (int)reason);
  fflush(0);
  return TRUE;
}
}
#else
# error oops!
#endif
