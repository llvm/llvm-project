#include <stdio.h>

int dummy() {
  printf("Dummy called\n");
  return 0;
}

int main(int argc, char **argv) {
  if (dummy() != 0)
    return 1;
  printf("Main called\n");
  return 0;
}
// Check that emitting trap value works properly and
// does not break functions
// REQUIRES: system-linux
// RUN: %clangxx -Wl,-q %s -o %t.exe
// RUN: %t.exe | FileCheck %s
// CHECK: Dummy called
// CHECK-NEXT: Main called
// RUN: llvm-bolt %t.exe -o %t.exe.bolt -lite=false --mark-funcs
// RUN: %t.exe.bolt | FileCheck %s
