// RUN: %clang_scudo %s -o %t
// RUN:     %run %t pointers 2>&1
// RUN:     %run %t contents 2>&1
// RUN: not %run %t memalign 2>&1 | FileCheck %s

// Tests that our reallocation function returns the same pointer when the
// requested size can fit into the previously allocated chunk. Also tests that
// a new chunk is returned if the size is greater, and that the contents of the
// chunk are left unchanged.
// As a final test, make sure that a chunk allocated by memalign cannot be
// reallocated.

#include <assert.h>
#include <malloc.h>
#include <string.h>

#include <vector>

int main(int argc, char **argv)
{
  void *p, *old_p;
  // Those sizes will exercise both allocators (Primary & Secondary).
  std::vector<size_t> sizes{1 << 5, 1 << 17};

  assert(argc == 2);
  for (size_t size : sizes) {
    if (!strcmp(argv[1], "pointers")) {
      old_p = p = realloc(nullptr, size);
      if (!p)
        return 1;
      size = malloc_usable_size(p);
      // Our realloc implementation will return the same pointer if the size
      // requested is lower or equal to the usable size of the associated chunk.
      p = realloc(p, size - 1);
      if (p != old_p)
        return 1;
      p = realloc(p, size);
      if (p != old_p)
        return 1;
      // And a new one if the size is greater.
      p = realloc(p, size + 1);
      if (p == old_p)
        return 1;
      // A size of 0 will free the chunk and return nullptr.
      p = realloc(p, 0);
      if (p)
        return 1;
      old_p = nullptr;
    }
    if (!strcmp(argv[1], "contents")) {
      p = realloc(nullptr, size);
      if (!p)
        return 1;
      for (int i = 0; i < size; i++)
        reinterpret_cast<char *>(p)[i] = 'A';
      p = realloc(p, size + 1);
      // The contents of the reallocated chunk must match the original one.
      for (int i = 0; i < size; i++)
        if (reinterpret_cast<char *>(p)[i] != 'A')
          return 1;
    }
    if (!strcmp(argv[1], "memalign")) {
      // A chunk coming from memalign cannot be reallocated.
      p = memalign(16, size);
      if (!p)
        return 1;
      p = realloc(p, size);
      free(p);
    }
  }
  return 0;
}

// CHECK: ERROR: invalid chunk type when reallocating address
