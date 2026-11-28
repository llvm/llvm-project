// Allocations that are too large for the size classes get their own mapping.
// When such a chunk leaves the quarantine, ASan has to get from the chunk
// header back to the start of the allocator's block. Check that this works for
// the header positions that different redzone sizes and alignments give.

// RUN: %clangxx_asan -O1 %s -o %t
// RUN: %env_asan_opts=quarantine_size_mb=0:thread_local_quarantine_size_kb=0 %run %t
// With a 16-byte redzone and an alignment of at most 16 the header is at the
// very start of the block.
// RUN: %env_asan_opts=quarantine_size_mb=0:thread_local_quarantine_size_kb=0:max_redzone=16 %run %t
// RUN: %env_asan_opts=quarantine_size_mb=0:thread_local_quarantine_size_kb=0:max_redzone=256 %run %t
// RUN: %env_asan_opts=quarantine_size_mb=1 %run %t
// RUN: %run %t
// RUN: not %run %t 0 2>&1 | FileCheck %s --check-prefix=BEG
// RUN: not %run %t 1 2>&1 | FileCheck %s --check-prefix=SIZE

// max_redzone=16 needs a small shadow granularity.
// REQUIRES: shadow-scale-3

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/asan_interface.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

// The page before the block starts with the address and size of the mapping.
__attribute__((no_sanitize("address"), noinline)) static void
CorruptMapping(void *p, int word) {
  uintptr_t page = sysconf(_SC_PAGESIZE);
  uintptr_t *header = (uintptr_t *)(((uintptr_t)p & ~(page - 1)) - page);
  header[word] = word ? 0 : (uintptr_t)p;
}

int main(int argc, char **argv) {
  if (argc > 1) {
    void *p = malloc(1 << 20);
    CorruptMapping(p, atoi(argv[1]));
    free(p);
    __sanitizer_purge_allocator();
    // BEG: CHECK failed: {{.*}} >= ((h->map_beg))
    // SIZE: CHECK failed: {{.*}} < ((h->map_beg + h->map_size))
    return 0;
  }
  const size_t kSizes[] = {130000, 131072, 200000, 1 << 20, (1 << 20) + 8};
  size_t heap_before = 0;
  for (int round = 0; round < 3; round++) {
    std::vector<void *> live;
    for (size_t size : kSizes) {
      for (size_t alignment = 8; alignment <= (1 << 18); alignment *= 2) {
        void *p = nullptr;
        int res = posix_memalign(&p, alignment, size);
        assert(!res);
        assert((uintptr_t)p % alignment == 0);
        assert(!__asan_region_is_poisoned(p, size));
        ((char *)p)[0] = ((char *)p)[size - 1] = 1;
        live.push_back(p);
      }
      live.push_back(malloc(size));
      live.push_back(calloc(1, size));
      // Freed here because the loops below use free().
      char *volatile array = new char[size];
      delete[] array;
    }
    // Free in an order that differs from the allocation order.
    for (size_t i = 0; i < live.size(); i += 2)
      free(live[i]);
    for (size_t i = 1; i < live.size(); i += 2)
      free(live[i]);
    // Empties the quarantine, so every large chunk has been unmapped by now.
    __sanitizer_purge_allocator();
    size_t heap = __sanitizer_get_heap_size();
    if (round)
      assert(heap == heap_before);
    heap_before = heap;
  }
  return 0;
}
