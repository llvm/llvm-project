// RUN: %clangxx_asan -coverage -O0 \
// RUN:   %if target={{.*-windows-msvc.*}} %{ -D_MT -D_DLL %} \
// RUN:   %if !MSVC && target={{.*-windows-msvc.*}} %{ \
// RUN:     -Wl,-nodefaultlib:libcmt,-defaultlib:msvcrt,-defaultlib:oldnames %} \
// RUN:   %s -o %t \
// RUN:   %if MSVC %{ /link /NODEFAULTLIB:libcmt /DEFAULTLIB:msvcrt /DEFAULTLIB:oldnames %}
// RUN: %env_asan_opts=check_initialization_order=1 %run %t 2>&1 | FileCheck %s

#include <stdio.h>
int foo() { return 1; }
int XXX = foo();
int main() {
  printf("PASS\n");
// CHECK: PASS
}
