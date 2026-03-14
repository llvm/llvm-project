// RUN: %clang_hwasan  %s -o %t
// RUN: %run %t 1 2>&1
// RUN: %run %t 2 2>&1

// REQUIRES: pointer-tagging

#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
  const size_t kPS = sysconf(_SC_PAGESIZE);
  const int kSize = kPS / atoi(argv[1]);
  const int kTag = 47;

  // Get any mmaped pointer.
  void *r =
      mmap(0, kSize, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (r == MAP_FAILED) {
    perror("Failed to mmap: ");
    abort();
  }

  // Check that the pointer is untagged.
  if (r != __hwasan_tag_pointer(r, 0)) {
    fprintf(stderr, "Pointer returned by mmap should be untagged.\n");
    abort();
  }

  // Manually tag the pointer and the memory.
  __hwasan_tag_memory(r, kTag, kPS);
  int *p1 = __hwasan_tag_pointer(r, kTag);

  // Check that the pointer and the tag match.
  if (__hwasan_test_shadow(p1, kPS) != -1) {
    fprintf(stderr, "Failed to tag.\n");
    abort();
  }

  if (munmap((char *)r + 1, kSize) == 0) {
    perror("munmap should fail: ");
    abort();
  }

  if (__hwasan_test_shadow(p1, kPS) != -1) {
    fprintf(stderr, "Still must be tagged.\n");
    abort();
  }

  // First munmmap and then mmap the same pointer using MAP_FIXED.
  if (munmap((char *)r, kSize) != 0) {
    perror("Failed to unmap: ");
    abort();
  }

  if (__hwasan_test_shadow(r, kPS) != -1) {
    fprintf(stderr, "Shadow memory was not cleaned by munmap.\n");
    abort();
  }
  __hwasan_tag_memory(r, kTag, kPS);
  int *p2 = (int *)mmap(r, kSize, PROT_READ | PROT_WRITE,
                        MAP_FIXED | MAP_ANON | MAP_PRIVATE, -1, 0);

  // Check that the pointer has no tag in it.
  if (p2 != r) {
    fprintf(stderr, "The mmap pointer has a non-zero tag in it.\n");
    abort();
  }

  // Make sure we can access the shadow with an untagged pointer.
  if (__hwasan_test_shadow(p2, kPS) != -1) {
    fprintf(stderr, "Shadow memory was not cleaned by mmap.\n");
    abort();
  }
  return 0;
}
