// Check that we report delete on a memory that belongs to a global variable.

// RUN: %clangxx_asan -g %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

static volatile char *x;
char a[10];

int main() {
  x = &a[0];
  delete x;
}

// CHECK: AddressSanitizer: attempting free on address which was not malloc()-ed
// CHECK: is located 0 bytes inside of global variable 'a' defined in
