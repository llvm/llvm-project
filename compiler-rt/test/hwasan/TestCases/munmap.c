// RUN: %clang_hwasan  %s -o %t
// RUN: %run %t 2>&1

// REQUIRES: aarch64-target-arch

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

int main(int argc, char **argv) {
  int size = 4096;
  int tag = 74;
  int val = 42;

  // Get any mmaped pointer.
  void *r =
      mmap(0, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (r == MAP_FAILED) {
    fprintf(stderr, "Failed to mmap.\n");
    abort();
  }

  // Check that the pointer is untagged.
  if (r != __hwasan_tag_pointer(r, 0)) {
    fprintf(stderr, "Pointer returned by mmap should be untagged.\n");
    return 1;
  }

  // Manually tag the pointer and the memory.
  __hwasan_tag_memory(r, tag, size);
  int *p1 = __hwasan_tag_pointer(r, tag);

  // Check that the pointer and the tag match.
  if (__hwasan_test_shadow(p1, size) != -1) {
    fprintf(stderr, "Failed to tag.\n");
    return 1;
  }

  // First munmmap and then mmap the same pointer using MAP_FIXED.
  munmap(r, size);
  if (__hwasan_test_shadow(r, size) != -1) {
    fprintf(stderr, "Shadow memory was not cleaned by munmap.\n");
    return 1;
  }
  __hwasan_tag_memory(r, tag, size);
  int *p2 = (int *)mmap(r, size, PROT_READ | PROT_WRITE,
                        MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);

  // Check that the pointer has no tag in it.
  if (p2 != r) {
    fprintf(stderr, "The mmap pointer has a non-zero tag in it.\n");
    return 1;
  }

  // Make sure we can access the shadow with an untagged pointer.
  if (__hwasan_test_shadow(p2, size) != -1 ) {
    fprintf(stderr, "Shadow memory was not cleaned by mmap.\n");
    return 1;
  }
  return 0;
}
