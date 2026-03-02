// Test that mmap with MAP_FIXED fails when attempting to overlap shadow memory.
//
// RUN: %clangxx_asan %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
//
// REQUIRES: !windows

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

extern "C" {
void __asan_get_shadow_mapping(unsigned long *shadow_scale,
                               unsigned long *shadow_offset);
}

int main() {
  unsigned long scale = 0;
  unsigned long offset = 0;
  __asan_get_shadow_mapping(&scale, &offset);

  if (offset == 0) {
    fprintf(stderr, "SKIPPED\n");
    return 0;
  }

  const size_t map_size = 4096;
  void *shadow_addr = (void *)offset;

  // MAP_FIXED on shadow memory should fail
  void *ptr = mmap(shadow_addr, map_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (ptr != MAP_FAILED || errno != EINVAL) {
    fprintf(stderr, "FAIL\n");
    return 1;
  }

  // MAP_FIXED in middle of shadow should fail
  void *mid_shadow = (void *)(offset + 0x100000);
  errno = 0;
  ptr = mmap(mid_shadow, map_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (ptr != MAP_FAILED || errno != EINVAL) {
    fprintf(stderr, "FAIL\n");
    return 1;
  }

  // Normal mmap should succeed
  errno = 0;
  ptr = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    fprintf(stderr, "FAIL\n");
    return 1;
  }
  munmap(ptr, map_size);

  fprintf(stderr, "PASS\n");
  return 0;
}
// CHECK: {{^PASS$|^SKIPPED$}}