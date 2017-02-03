// RUN: %clang_scudo %s -lstdc++ -o %t
// RUN: %run %t 2>&1

// Tests that a regular workflow of allocation, memory fill and free works as
// intended. Tests various sizes serviced by the primary and secondary
// allocators.

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

int main(int argc, char **argv)
{
  void *p;
  std::vector<ssize_t> sizes{1, 8, 16, 32, 1024, 32768,
    1 << 16, 1 << 17, 1 << 20, 1 << 24};
  std::vector<int> offsets{1, 0, -1, -7, -8, -15, -16, -31, -32};

  p = malloc(0);
  assert(p);
  free(p);
  for (ssize_t size : sizes) {
    for (int offset: offsets) {
      ssize_t actual_size = size + offset;
      if (actual_size <= 0)
        continue;
      p = malloc(actual_size);
      assert(p);
      memset(p, 0xff, actual_size);
      free(p);
    }
  }

  return 0;
}
