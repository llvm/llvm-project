// RUN: %clang_dfsan %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %run %t foo
//
// REQUIRES: x86_64-target-arch

#include <stdio.h>

int do_nothing(const char *format, ...) {
  return 0;
}

int main(int argc, char **argv) {
  int (*fp)(const char *, ...);

  if (argc > 1)
    fp = do_nothing;
  else
    fp = printf;

  // CHECK: FATAL: DataFlowSanitizer: unsupported indirect call to vararg function printf
  fp("hello %s\n", "world");
}
