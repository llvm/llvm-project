// Test malloc alignment.
// RUN: %clang_hwasan -mllvm -hwasan-globals=0 %s -o %t
// RUN: %run %t

#include <assert.h>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>

static const size_t sizes[] = {
    1, 3, 7, 8, 9, 16, 17, 31, 32, 33,
    63, 64, 65, 127, 128, 129, 511, 512, 513, 2047,
    2048, 2049, 65535, 65536, 65537, 1048575, 1048576, 1048577};
static const size_t alignments[] = {8, 16, 64, 256, 1024, 4096, 65536, 131072};

__attribute__((no_sanitize("hwaddress"))) int main() {
  for (unsigned i = 0; i < sizeof(sizes) / sizeof(*sizes); ++i) {
    for (unsigned j = 0; j < sizeof(alignments) / sizeof(*alignments); ++j) {
      size_t size = sizes[i];
      size_t alignment = alignments[j];
      fprintf(stderr, "size %zu, alignment %zu (0x%zx)\n", size, alignment,
              alignment);
      const int cnt = 10;
      void *ptrs[cnt];
      for (int k = 0; k < cnt; ++k) {
        int res = posix_memalign(&ptrs[k], alignment, size);
        assert(res == 0);
        fprintf(stderr, "... addr 0x%zx\n", (size_t)ptrs[k]);
        assert(((size_t)ptrs[k] & (alignment - 1)) == 0);
      }
      for (int k = 0; k < cnt; ++k)
        free(ptrs[k]);
    }
  }
  return 0;
}
