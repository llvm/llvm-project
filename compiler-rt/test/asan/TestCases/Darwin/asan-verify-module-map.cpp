// This test simply checks that the "Invalid module map" warning is not printed
// in the output of a backtrace.

// RUN: %clangxx_asan -O0 -g %s -o %t.executable
// RUN: %env_asan_opts="print_module_map=2" not %run %t.executable 2>&1 | FileCheck %s

// CHECK-NOT: WARN: Invalid module map

#include <cstdlib>

extern "C" void foo(int *a) { *a = 5; }

int main() {
  int *a = (int *)malloc(sizeof(int));
  if (!a)
    return 0;
  free(a);
  foo(a);
  return 0;
}