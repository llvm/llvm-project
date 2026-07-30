// If we have LLD, see that things more or less work.
//
// REQUIRES: lld-available
//
// RUN: %clangxx_asan %if target={{.*-windows-gnu}} %{ -gcodeview -gcolumn-info -Wl,--pdb= %} %else %{ -g -gcodeview -Wl,-debug %} -O2 %s -o %t.exe -fuse-ld=lld
// RUN: not %run %t.exe 2>&1 | FileCheck %s --check-prefixes=CHECK,%if target={{.*-windows-msvc.*}} %{MSVC%} %else %{MINGW%}

#include <stdlib.h>

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: heap-use-after-free
  // CHECK: free
  // MSVC: main{{.*}}fuse-lld.cpp:[[@LINE-4]]:3
  // MINGW: {{.*fuse-lld.cpp.tmp.exe\+0x14000[0-9a-f]+}}
  // CHECK: malloc
  // MSVC: main{{.*}}fuse-lld.cpp:[[@LINE-8]]:20
  // MINGW: {{.*fuse-lld.cpp.tmp.exe\+0x14000[0-9a-f]+}}
}
