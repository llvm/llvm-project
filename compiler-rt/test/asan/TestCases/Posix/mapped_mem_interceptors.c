// Test for mmap/munmap interceptors.
// RUN: %clang_asan  %s -o %t
// RUN: %run %t 2>&1

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

int main(int argc, char **argv) {
  int size = 4096;
  int val = 42;

  // Get any mmaped pointer.
  void *r =
      mmap(0, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  assert(r != MAP_FAILED);

  // Make sure the memory is unpoisoned.
  if (__asan_region_is_poisoned(r, size) != 0) {
    fprintf(stderr, "Memory returned by mmap should be unpoisoned.\n");
    abort();
  }

  // First munmmap and then mmap the same pointer using MAP_FIXED.
  __asan_poison_memory_region(r, size);
  munmap(r, size);
  if (__asan_region_is_poisoned(r, size) != 0) {
    fprintf(stderr, "Shadow memory was not cleaned by munmap.\n");
    abort();
  }
  __asan_poison_memory_region(r, size);
  void *p = mmap(r, size, PROT_READ | PROT_WRITE,
                 MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);
  assert(r == p);

  // Make sure the memory is unpoisoned.
  if (__asan_region_is_poisoned(r, size) != 0) {
    fprintf(stderr, "Memory returned by mmap should be unpoisoned.\n");
    abort();
  }

  return 0;
}
