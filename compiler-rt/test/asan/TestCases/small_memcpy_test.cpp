// Test that small memcpy works correctly.

// RUN: %clangxx_asan %s -o %t
// RUN: not %run %t 8 24 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: not %run %t 16 32 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: not %run %t 24 40 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: not %run %t 32 48 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: not %run %t 40 56 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: not %run %t 48 64 2>&1 | FileCheck %s --check-prefix=CHECK
// REQUIRES: shadow-scale-3
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  assert(argc == 3);
  size_t poison_from = atoi(argv[1]);
  size_t poison_to = atoi(argv[2]);
  assert(poison_from <= poison_to);
  char A1[64], A2[64];
  fprintf(stderr, "%zd %zd\n", poison_from, poison_to - poison_from);
  __asan_poison_memory_region(&A1[0] + poison_from, poison_to - poison_from);
  memcpy(A1, A2, sizeof(A1));
// CHECK: AddressSanitizer: use-after-poison
  return 0;
}
