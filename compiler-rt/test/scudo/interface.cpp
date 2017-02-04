// RUN: %clang_scudo %s -lstdc++ -o %t
// RUN: %run %t ownership          2>&1
// RUN: %run %t ownership-and-size 2>&1
// RUN: %run %t heap-size          2>&1

// Tests that the sanitizer interface functions behave appropriately.

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <vector>

#include <sanitizer/allocator_interface.h>

int main(int argc, char **argv)
{
  assert(argc == 2);

  if (!strcmp(argv[1], "ownership")) {
    // Ensures that __sanitizer_get_ownership can be called before any other
    // allocator function, and that it behaves properly on a pointer not owned
    // by us.
    assert(!__sanitizer_get_ownership(argv));
  }
  if (!strcmp(argv[1], "ownership-and-size")) {
    // Tests that __sanitizer_get_ownership and __sanitizer_get_allocated_size
    // behave properly on chunks allocated by the Primary and Secondary.
    void *p;
    std::vector<ssize_t> sizes{1, 8, 16, 32, 1024, 32768,
      1 << 16, 1 << 17, 1 << 20, 1 << 24};
    for (size_t size : sizes) {
      p = malloc(size);
      assert(p);
      assert(__sanitizer_get_ownership(p));
      assert(__sanitizer_get_allocated_size(p) >= size);
      free(p);
    }
  }
  if (!strcmp(argv[1], "heap-size")) {
    // Ensures that __sanitizer_get_heap_size can be called before any other
    // allocator function.
    assert(__sanitizer_get_heap_size() >= 0);
  }

  return 0;
}
