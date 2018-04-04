// RUN: %clang_noscs %s -o %t
// RUN: %run %t 3 | FileCheck %s
// RUN: %run %t 12 | FileCheck -check-prefix=OVERFLOW_SUCCESS %s

// RUN: %clang_scs %s -o %t
// RUN: %run %t 3 | FileCheck %s
// RUN: not --crash %run %t 12

// Test that a stack overflow fails as expected

#include <stdio.h>
#include <stdlib.h>

#include "minimal_runtime.h"

void print_and_exit(void) {
// CHECK-NOT: Stack overflow successful.
// OVERFLOW_SUCCESS: Stack overflow successful.
  printf("Stack overflow successful.\n");
  exit(0);
}

int main(int argc, char **argv)
{
  if (argc != 2)
    exit(1);

  void *addrs[4];
  const int iterations = atoi(argv[1]);
  for (int i = 0; i < iterations; i++)
    addrs[i] = &print_and_exit;

  printf("Returning.\n");

  return 0;
}
