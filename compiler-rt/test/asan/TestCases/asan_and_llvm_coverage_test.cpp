// RUN: %clangxx_asan -coverage -O0 \
// RUN:   %if target={{.*-windows-msvc.*}} %{ \
// RUN:     -D_MT -D_DLL \
// RUN:     -Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames \
// RUN:   %} %s -o %t
// RUN: %env_asan_opts=check_initialization_order=1 %run %t 2>&1 | FileCheck %s

#include <stdio.h>
int foo() { return 1; }
int XXX = foo();
int main() {
  printf("PASS\n");
// CHECK: PASS
}
