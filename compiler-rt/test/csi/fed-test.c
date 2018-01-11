// RUN: %clang_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_csi_toolc %tooldir/fed-test-tool.c -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_csi_c %s -o %t.o
// RUN: %clang_csi %t.o %t-tool.o %csirtlib -o %t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

static void foo() {
  printf("In foo.\n");
}

int main(int argc, char **argv) {
  printf("In main.\n");
  foo();
  // CHECK: Enter function 0 [{{.*}}fed-test.c:14]
  // CHECK: Enter function 1 [{{.*}}fed-test.c:10]
  return 0;
}
