// Check that we report delete on a memory that belongs to a stack variable.

// RUN: %clangxx_asan -g %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

static volatile char *x;

int main() {
  char a[10];
  x = &a[0];
  delete x;
}

// CHECK: AddressSanitizer: attempting free on address which was not malloc()-ed
// CHECK: is located in stack of thread T0 at offset
// CHECK: 'a'{{.*}} <== Memory access at offset {{16|32}} is inside this variable
