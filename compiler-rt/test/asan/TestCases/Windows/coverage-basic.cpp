// RUN: rm -rf %t-dir
// RUN: mkdir %t-dir && cd %t-dir
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s -o test.exe
// RUN: %env_asan_opts=coverage=1 %run ./test.exe
//
// RUN: %sancov print *.sancov | FileCheck %s

// FIXME: Investigate failure on MinGW.
// XFAIL: target={{.*-windows-gnu}}

#include <stdio.h>

void foo() { fputs("FOO", stderr); }
void bar() { fputs("BAR", stderr); }

int main(int argc, char **argv) {
  if (argc == 2) {
    foo();
    bar();
  } else {
    bar();
    foo();
  }
}

// CHECK: 0x{{[0-9a-f]*}}
// CHECK: 0x{{[0-9a-f]*}}
// CHECK: 0x{{[0-9a-f]*}}
// CHECK-NOT: 0x{{[0-9a-f]*}}
