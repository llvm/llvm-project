// RUN: %clang_scudo %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// Tests that the allocator stats printing function exists and outputs
// "something". Currently that "something" is fairly nebulous, as the 32-bit
// primary doesn't output anything, and for the 64-bit one it's highly dependent
// on the size class map and potential library allocations. So keep it very
// generic for now.

#include <stdlib.h>

#include <sanitizer/scudo_interface.h>

int main(int argc, char **argv) {
  free(malloc(1U));
  __scudo_print_stats();
  return 0;
}

// CHECK: Stats:
