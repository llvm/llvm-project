// RUN: %clangxx_asan -coverage -O0 %s -o %t
// RUN: %env_asan_opts=check_initialization_order=1 %run %t 2>&1 | FileCheck %s

// We don't really support running tests using profile runtime on Windows.
// UNSUPPORTED: win32
#include <stdio.h>
int foo() { return 1; }
int XXX = foo();
int main() {
  printf("PASS\n");
// CHECK: PASS
}
