// Make sure everything works even if the main module doesn't have any stack
// variables, thus doesn't explicitly reference any symbol exported by the
// runtime thunk.
//
// RUN: %clang_cl_asan %LD %Od -DDLL1 %s %Fe%t1.dll \
// RUN:   %if target={{.*-windows-gnu}} %{ -Wl,--out-implib,%t1.lib %}
// RUN: %clang_cl_asan %LD %Od -DDLL2 %s %Fe%t2.dll \
// RUN:   %if target={{.*-windows-gnu}} %{ -Wl,--out-implib,%t2.lib %}
// RUN: %clang_cl_asan %Od -DEXE %s %t1.lib %t2.lib %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <malloc.h>
#include <string.h>

extern "C" {
#if defined(EXE)
__declspec(dllimport) void foo1();
__declspec(dllimport) void foo2();

int main() {
  foo1();
  foo2();
}
#elif defined(DLL1)
__declspec(dllexport) void foo1() {}
#elif defined(DLL2)
__attribute__((noinline))
static void NullDeref(int *ptr) {
  // CHECK: ERROR: AddressSanitizer: access-violation on unknown address
  // CHECK:   {{0x0*000.. .*pc 0x.*}}
  ptr[10]++;  // BOOM
}

__declspec(dllexport) void foo2() {
  NullDeref((int*)0);
  // CHECK: {{    #1 0x.* in foo2.*null_deref_multiple_dlls.cpp:}}[[@LINE-1]]
  // CHECK: AddressSanitizer can not provide additional info.
}
#else
# error oops!
#endif
}
