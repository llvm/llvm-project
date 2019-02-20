// Check that we fill malloc-ed memory correctly.
// RUN: %clangxx_hwasan %s -o %t
// RUN: %run %t | FileCheck %s
// RUN: %env_hwasan_opts=max_malloc_fill_size=10:malloc_fill_byte=8 %run %t | FileCheck %s --check-prefix=CHECK-10-8
// RUN: %env_hwasan_opts=max_malloc_fill_size=20:malloc_fill_byte=171 %run %t | FileCheck %s --check-prefix=CHECK-20-ab

#include <stdio.h>
int main(int argc, char **argv) {
  // With asan allocator this makes sure we get memory from mmap.
  static const int kSize = 1 << 25;
  unsigned char *x = new unsigned char[kSize];
  printf("-");
  for (int i = 0; i <= 32; i++) {
    printf("%02x", x[i]);
  }
  printf("-\n");
  delete [] x;
}

// CHECK: -bebebebebebebebebebebebebebebebebebebebebebebebebebebebebebebebebe-
// CHECK-10-8: -080808080808080808080000000000000000000000000000000000000000000000-
// CHECK-20-ab: -abababababababababababababababababababab00000000000000000000000000-
