// When we link a binary without the -debug flag, ASan should print out VAs
// instead of RVAs. The frames for main and do_uaf should be above 0x400000,
// which is the default image base of an executable.

// RUN: rm -f %t.pdb
// RUN: %if target={{.*-windows-gnu}} %{ \
// RUN:   %clangxx_asan -O2 %s -o %t.exe -gcodeview -gcolumn-info -fuse-ld=lld \
// RUN: %} %else %{ \
// RUN:   %clangxx_asan -c -O2 %s -o %t.obj && \
// RUN:   lld-link /nologo /OUT:%t.exe %t.obj -defaultlib:libcmt -nodefaultlib:msvcrt -defaultlib:oldnames %asan_static_runtime_thunk %asan_lib \
// RUN: %}
// RUN: not %run %t.exe 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-msvc.*}} %{MSVC%} %else %{MINGW%}
// REQUIRES: lld-available

#include "../defines.h"
#include <stdio.h>
#include <stdlib.h>
int ATTRIBUTE_NOINLINE do_uaf(void);
int main() {
  int r = do_uaf();
  printf("r: %d\n", r);
  return r;
}
int do_uaf(void) {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: AddressSanitizer: heap-use-after-free
  // MSVC: #0 {{0x[a-f0-9]+ \(.*[\\/]unsymbolized.cpp.*.exe\+(0x40|0x14000)[a-f0-9]{4}\)}}
  // MSVC: #1 {{0x[a-f0-9]+ \(.*[\\/]unsymbolized.cpp.*.exe\+(0x40|0x14000)[a-f0-9]{4}\)}}
  // MINGW: #0 {{0x[a-f0-9]+ in do_uaf\(\) \(.*[\\/]unsymbolized.cpp.*.exe\+0x14000[a-f0-9]{4}\)}}
  // MINGW: #1 {{0x[a-f0-9]+ in main \(.*[\\/]unsymbolized.cpp.*.exe\+0x14000[a-f0-9]{4}\)}}
}
