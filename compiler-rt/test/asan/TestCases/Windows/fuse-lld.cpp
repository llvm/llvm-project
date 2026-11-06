// If we have LLD, see that things more or less work.
//
// REQUIRES: lld-available
//
// RUN: %clangxx_asan %if target={{.*-windows-gnu}} %{ -gcodeview -gcolumn-info -Wl,--pdb= -Wl,-S %} %else %{ -g -gcodeview -Wl,-debug %} -O2 %s -o %t.exe -fuse-ld=lld
// RUN: not %run %t.exe 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *x = (char*)malloc(10 * sizeof(char));
  free(x);
  return x[5];
  // CHECK: heap-use-after-free
  // CHECK: free
  // CHECK: main{{.*}}fuse-lld.cpp:[[@LINE-4]]:3
  // CHECK: malloc
  // CHECK: main{{.*}}fuse-lld.cpp:[[@LINE-7]]:20
}
