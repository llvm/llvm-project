// RUN: %clangxx -O0 -g %s -o %t && %run %t

// UBSan does not have its own allocator
// UNSUPPORTED: ubsan

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Based on lib/msan/tests/msan_test.cpp::get_allocated_size_and_ownership
int main(void) {
  int sizes[] = {10, 100, 1000, 10000, 100000, 1000000};

  for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {
    printf("Testing size %d\n", sizes[i]);

    char *array = reinterpret_cast<char *>(malloc(sizes[i]));
    int *int_ptr = new int;
    printf("array: %p\n", array);
    printf("int_ptr: %p\n", int_ptr);

    // Bogus value to unpoison start. Calling __sanitizer_get_allocated_begin
    // does not unpoison it.
    const void *start = NULL;
    for (int j = 0; j < sizes[i]; j++) {
      printf("j: %d\n", j);

      start = __sanitizer_get_allocated_begin(array + j);
      printf("Start: %p (expected: %p)\n", start, array);
      fflush(stdout);
      assert(array == start);
    }

    start = __sanitizer_get_allocated_begin(int_ptr);
    assert(int_ptr == start);

    void *wild_addr = reinterpret_cast<void *>(4096 * 160);
    assert(__sanitizer_get_allocated_begin(wild_addr) == NULL);

    wild_addr = reinterpret_cast<void *>(0x1);
    assert(__sanitizer_get_allocated_begin(wild_addr) == NULL);

    // NULL is a valid argument for GetAllocatedSize but is not owned.
    assert(__sanitizer_get_allocated_begin(NULL) == NULL);

    free(array);
    for (int j = 0; j < sizes[i]; j++) {
      assert(__sanitizer_get_allocated_begin(array + j) == NULL);
    }

    delete int_ptr;
    assert(__sanitizer_get_allocated_begin(int_ptr) == NULL);
  }

  return 0;
}
