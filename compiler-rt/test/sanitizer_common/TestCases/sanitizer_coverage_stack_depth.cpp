// Tests -fsanitize-coverage=stack-depth
//
// RUN: %clangxx -O0 -std=c++11 -fsanitize-coverage=stack-depth %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not Assertion{{.*}}failed
// RUN: %clangxx -O0 -std=c++11 -fsanitize-coverage=trace-pc-guard,stack-depth \
// RUN:     %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not Assertion{{.*}}failed

#include <cstdint>
#include <cstdio>
#include <cassert>

thread_local uintptr_t __sancov_lowest_stack;
uintptr_t last_stack;

void foo(int recurse) {
  assert(__sancov_lowest_stack < last_stack);
  last_stack = __sancov_lowest_stack;
  if (recurse <= 0) return;
  foo(recurse - 1);
}

int main() {
  last_stack = __sancov_lowest_stack;
  foo(100);
  printf("Success!\n");
  return 0;
}

// CHECK: Success!
