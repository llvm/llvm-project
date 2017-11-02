// RUN: %clangxx_scudo %s -lstdc++ -o %t
// RUN: %run %t pointers 2>&1
// RUN: %run %t contents 2>&1
// RUN: %run %t usablesize 2>&1

// Tests that our reallocation function returns the same pointer when the
// requested size can fit into the previously allocated chunk. Also tests that
// a new chunk is returned if the size is greater, and that the contents of the
// chunk are left unchanged. Finally, checks that realloc copies the usable
// size of the old chunk to the new one (as opposed to the requested size).

#include <assert.h>
#include <malloc.h>
#include <string.h>

#include <vector>

int main(int argc, char **argv)
{
  void *p, *old_p;
  // Those sizes will exercise both allocators (Primary & Secondary).
  std::vector<size_t> sizes{1, 16, 1024, 32768, 1 << 16, 1 << 17, 1 << 20};

  assert(argc == 2);

  if (!strcmp(argv[1], "usablesize")) {
    // This tests a sketchy behavior inherited from poorly written libraries
    // that have become somewhat standard. When realloc'ing a chunk, the
    // copied contents should span the usable size of the chunk, not the
    // requested size.
    size_t size = 496, usable_size;
    p = nullptr;
    // Make sure we get a chunk with a usable size actually larger than size.
    do {
      if (p) free(p);
      size += 16;
      p = malloc(size);
      usable_size = malloc_usable_size(p);
      assert(usable_size >= size);
    } while (usable_size == size);
    for (int i = 0; i < usable_size; i++)
      reinterpret_cast<char *>(p)[i] = 'A';
    old_p = p;
    // Make sure we get a different chunk so that the data is actually copied.
    do {
      size *= 2;
      p = realloc(p, size);
      assert(p);
    } while (p == old_p);
    // The contents of the new chunk must match the old one up to usable_size.
    for (int i = 0; i < usable_size; i++)
      assert(reinterpret_cast<char *>(p)[i] == 'A');
    free(p);
  } else {
    for (size_t size : sizes) {
      if (!strcmp(argv[1], "pointers")) {
        old_p = p = realloc(nullptr, size);
        assert(p);
        size = malloc_usable_size(p);
        // Our realloc implementation will return the same pointer if the size
        // requested is lower than or equal to the usable size of the associated
        // chunk.
        p = realloc(p, size - 1);
        assert(p == old_p);
        p = realloc(p, size);
        assert(p == old_p);
        // And a new one if the size is greater.
        p = realloc(p, size + 1);
        assert(p != old_p);
        // A size of 0 will free the chunk and return nullptr.
        p = realloc(p, 0);
        assert(!p);
        old_p = nullptr;
      }
      if (!strcmp(argv[1], "contents")) {
        p = realloc(nullptr, size);
        assert(p);
        for (int i = 0; i < size; i++)
          reinterpret_cast<char *>(p)[i] = 'A';
        p = realloc(p, size + 1);
        // The contents of the reallocated chunk must match the original one.
        for (int i = 0; i < size; i++)
          assert(reinterpret_cast<char *>(p)[i] == 'A');
      }
    }
  }
  return 0;
}

// CHECK: ERROR: invalid chunk type when reallocating address
