// RUN: %clang_csi_toolc %tooldir/null-tool.c -o %t-null-tool.o
// RUN: %clang_csi_toolc %tooldir/load-property-test-tool.c -o %t-tool.o
// RUN: %link_csi %t-tool.o %t-null-tool.o -o %t-tool.o
// RUN: %clang_csi_c %s -o %t.o
// RUN: %clang_csi %t.o %t-tool.o -o %t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

static int global = 0;

int main(int argc, char **argv) {
  int x = global + 1;               // Read of global
  x++;                              // Read-before-write of x
  printf("x is %d\n", x);           // Function calls partition read-before-write analysis
  x += global++;                    // Read-before-write of global and x */
  printf("x is %d\n", x);           // Read on x
  printf("global is %d\n", global); // Read on global
  // CHECK: num_loads = 7
  // CHECK: num_read_before_writes = 3
  return 0;
}
