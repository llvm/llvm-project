// RUN: rm -rf %T/coverage-basic
// RUN: mkdir %T/coverage-basic && cd %T/coverage-basic
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o test.exe
// RUN: %env_asan_opts=coverage=1 %run ./test.exe
//
// RUN: %sancov print *.sancov | FileCheck %s
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
