// Test that the raw binary profile is correctly generated on Darwin.
// Verify the magic number header to ensure profile serialization works.

// RUN: %clangxx_memprof %s -o %t
// RUN: %env_memprof_opts=log_path=stdout %run %t > %t.memprofraw
// RUN: od -c -N 8 %t.memprofraw | FileCheck %s

#include <cstdlib>
#include <cstring>

int main() {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}

// Check the raw profile magic number (little-endian).
// CHECK: 0000000 201   r   f   o   r   p   m 377
