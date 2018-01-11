// RUN: %clang_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_csi_toolc %tooldir/function-call-count-tool.c -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_csi_c -fPIC %supportdir/libtest.c -o %t-libtest.o
// RUN: %clang_csi -Wl,-soname,libtest.so -shared %t-libtest.o %t-tool.o -o %T/libtest.so
// RUN: %clang_csi_c %s -o %t.o
// RUN: %clang_csi -Wl,-rpath,%T -L %T %t.o %t-tool.o -ltest %csirtlib -o %t
// RUN: LD_LIBRARY_PATH=%T:$LD_LIBRARY_PATH %run %t | FileCheck %s

#include <stdio.h>
#include "support/libtest.h"

int main(int argc, char **argv) {
  printf("One call.\n");
  printf("Two calls.\n");
  libtest();
  // CHECK: num_function_calls = 4
  return 0;
}
