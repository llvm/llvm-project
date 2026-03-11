// Test that munmap correctly handles application memory regions.
//
// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 2>&1
//
// REQUIRES: !windows

#include <errno.h>
#include <stdio.h>
#include <sys/mman.h>

int main() {
  const size_t map_size = 8192;

  // Map application memory
  void *ptr = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "ERROR: mmap failed\n");
    return 1;
  }

  // Normal munmap should succeed
  int res = munmap(ptr, map_size);
  if (res != 0) {
    fprintf(stderr, "ERROR: munmap failed with errno=%d\n", errno);
    return 1;
  }

  fprintf(stderr, "PASS\n");
  // CHECK: PASS
  return 0;
}