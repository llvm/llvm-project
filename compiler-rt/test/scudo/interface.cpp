// RUN: %clang_scudo %s -o %t
// RUN: %run %t 2>&1

// Tests that the sanitizer interface functions behave appropriately.

#include <stdlib.h>

#include <vector>

#include <sanitizer/allocator_interface.h>

int main(int argc, char **argv)
{
  void *p;
  std::vector<ssize_t> sizes{1, 8, 16, 32, 1024, 32768,
    1 << 16, 1 << 17, 1 << 20, 1 << 24};
  for (size_t size : sizes) {
    p = malloc(size);
    if (!p)
      return 1;
    if (!__sanitizer_get_ownership(p))
      return 1;
    if (__sanitizer_get_allocated_size(p) < size)
      return 1;
    free(p);
  }
  return 0;
}
