/// From sanitizer_common/TestCases/allocator_interface.cpp
// RUN: %clangxx_nsan %s -o %t && %run %t 1234
// RUN: %clangxx_nsan %s -o %t && %run %t 5678910

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

void Test(int size) {
  auto allocated_bytes_before = __sanitizer_get_current_allocated_bytes();
  int *p = (int *)malloc(size);
  assert(__sanitizer_get_estimated_allocated_size(size) >= size);
  assert(__sanitizer_get_ownership(p));
  assert(!__sanitizer_get_ownership(&p));
  assert(__sanitizer_get_allocated_size(p) == size);
  assert(__sanitizer_get_allocated_size_fast(p) == size);
  assert(__sanitizer_get_allocated_begin(p) == p);
  assert(__sanitizer_get_allocated_begin(p + 1) == p);
  assert(__sanitizer_get_current_allocated_bytes() >=
         size + allocated_bytes_before);
  assert(__sanitizer_get_current_allocated_bytes() <=
         2 * size + allocated_bytes_before);
  assert(__sanitizer_get_heap_size() >= size);
  free(p);

  // These are not implemented.
  assert(__sanitizer_get_unmapped_bytes() <= 1);
  assert(__sanitizer_get_free_bytes() > 0);

  __sanitizer_purge_allocator();
}

int main(int argc, char **argv) {
  int size = atoi(argv[1]);

  Test(size);

  // Check the thread local caches work as well.
  std::thread t(Test, size);
  t.join();

  return 0;
}
