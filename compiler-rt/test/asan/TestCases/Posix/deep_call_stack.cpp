// Check that UAR mode can handle very deep recusrion.
// REQUIRES: shell
// RUN: %clangxx_asan -O2 %s -o %t
// RUN: ulimit -s 4096
// RUN: %env_asan_opts=detect_stack_use_after_return=1 %run %t 2>&1 | FileCheck %s

// Also check that use_sigaltstack+verbosity doesn't crash.
// RUN: %env_asan_opts=verbosity=1:use_sigaltstack=1:detect_stack_use_after_return=1 %run %t  | FileCheck %s

// UNSUPPORTED: ios

#include <stdio.h>

__attribute__((noinline))
void RecursiveFunc(int depth, int *ptr) {
  if ((depth % 1000) == 0)
    printf("[%05d] ptr: %p\n", depth, ptr);
  if (depth == 0)
    return;
  int local;
  RecursiveFunc(depth - 1, &local);
}

int main(int argc, char **argv) {
  RecursiveFunc(15000, 0);
  return 0;
}
// CHECK: [15000] ptr:
// CHECK: [07000] ptr:
// CHECK: [00000] ptr:
