// Check that we report new[] vs delete as alloc-dealloc-mismatch and not as
// new-delete-type-mismatch when -fsized-deallocation is enabled.

// RUN: %clangxx_asan -g %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -fsized-deallocation -g %s -o %t && %env_asan_opts=alloc_dealloc_mismatch=1 not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

static volatile char *x;

int main() {
  x = new char[10];
  delete x;
}

// CHECK: AddressSanitizer: alloc-dealloc-mismatch (operator new [] vs operator delete) on 0x
