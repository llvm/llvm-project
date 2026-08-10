// Ensure that output is the same but exit code depends on halt_on_error value
// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts="halt_on_error=0" %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts="halt_on_error=1" not %run %t 2>&1 | FileCheck %s
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: leak-detection
// UNSUPPORTED: android 

#include <stdlib.h>

int f() {
  volatile int *a = (int *)malloc(20);
  a[0] = 1;
  return a[0];
}

int main() {
  f();
  f();
}

// CHECK: LeakSanitizer: detected memory leaks
