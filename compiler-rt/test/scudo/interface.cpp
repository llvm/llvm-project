// RUN: %clangxx_scudo %s -lstdc++ -o %t
// RUN:                                                   %run %t ownership          2>&1
// RUN:                                                   %run %t ownership-and-size 2>&1
// RUN:                                                   %run %t heap-size          2>&1
// RUN: %env_scudo_opts="allocator_may_return_null=1"     %run %t soft-limit         2>&1
// RUN: %env_scudo_opts="allocator_may_return_null=1" not %run %t hard-limit         2>&1

// Tests that the sanitizer interface functions behave appropriately.

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>

#include <vector>

#include <sanitizer/allocator_interface.h>
#include <sanitizer/scudo_interface.h>

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
  if (!strcmp(argv[1], "soft-limit")) {
    // Verifies that setting the soft RSS limit at runtime works as expected.
    std::vector<void *> pointers;
    size_t size = 1 << 19;  // 512Kb
    for (int i = 0; i < 5; i++) {
      void *p = malloc(size);
      memset(p, 0, size);
      pointers.push_back(p);
    }
    // Set the soft RSS limit to 1Mb.
    __scudo_set_rss_limit(1, 0);
    usleep(20000);
    // The following allocation should return NULL.
    void *p = malloc(size);
    assert(!p);
    // Remove the soft RSS limit.
    __scudo_set_rss_limit(0, 0);
    // The following allocation should succeed.
    p = malloc(size);
    assert(p);
    free(p);
    while (!pointers.empty()) {
      free(pointers.back());
      pointers.pop_back();
    }
  }
  if (!strcmp(argv[1], "hard-limit")) {
    // Verifies that setting the hard RSS limit at runtime works as expected.
    std::vector<void *> pointers;
    size_t size = 1 << 19;  // 512Kb
    for (int i = 0; i < 5; i++) {
      void *p = malloc(size);
      memset(p, 0, size);
      pointers.push_back(p);
    }
    // Set the hard RSS limit to 1Mb
    __scudo_set_rss_limit(1, 1);
    usleep(20000);
    // The following should trigger our death.
    void *p = malloc(size);
  }

  return 0;
}
