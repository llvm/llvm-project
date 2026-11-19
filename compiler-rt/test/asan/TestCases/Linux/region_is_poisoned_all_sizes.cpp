// Checks __asan_region_is_poisoned for start offsets 0 to 17 and every size
// that fits in 1400 bytes, with nothing poisoned and with one poisoned spot at
// various positions.

// RUN: %clangxx_asan -O1 %s -o %t && %run %t
// RUN: not %run %t wrap 2>&1 | FileCheck %s --check-prefix=WRAP

// The poisoned spots end at a multiple of 8 bytes.
// REQUIRES: shadow-scale-3

#include <assert.h>
#include <sanitizer/asan_interface.h>
#include <stdio.h>
#include <stdlib.h>

static const size_t kSize = 1400;

static void Check(char *buf, size_t poisoned_beg, size_t poisoned_end) {
  for (size_t beg = 0; beg < 18; beg++) {
    for (size_t size = 0; beg + size <= kSize; size++) {
      size_t first = beg > poisoned_beg ? beg : poisoned_beg;
      size_t limit = beg + size < poisoned_end ? beg + size : poisoned_end;
      void *expected = first < limit ? buf + first : nullptr;
      void *actual = __asan_region_is_poisoned(buf + beg, size);
      if (actual != expected) {
        fprintf(stderr, "poisoned [%zu, %zu), region [%zu, %zu): %p vs %p\n",
                poisoned_beg, poisoned_end, beg, beg + size, actual, expected);
        abort();
      }
    }
  }
}

int main(int argc, char **argv) {
  char *buf = (char *)malloc(kSize);
  if (argc > 1) {
    // The region wraps around the end of the address space.
    // WRAP: CHECK failed: {{.*}}asan_poisoning.cpp{{.*}}((beg)) <= ((last))
    __asan_region_is_poisoned(buf, (size_t)-8);
    return 0;
  }
  assert(((size_t)buf & 7) == 0);
  Check(buf, 0, 0);
  for (size_t pos = 0; pos < kSize; pos += pos < 160 ? 1 : 37) {
    // Poisoning up to the end of an 8-byte granule is exact.
    size_t end = (pos | 7) + 1;
    __asan_poison_memory_region(buf + pos, end - pos);
    Check(buf, pos, end);
    __asan_unpoison_memory_region(buf + pos, end - pos);
  }
  Check(buf, 0, 0);

  // The byte after the allocation is in the redzone.
  assert(__asan_region_is_poisoned(buf, kSize + 1) == buf + kSize);
  assert(__asan_region_is_poisoned(buf + kSize - 100, 101) == buf + kSize);
  // So is the byte in front of it.
  assert(__asan_region_is_poisoned(buf - 1, 200) == buf - 1);

  // A region that starts outside application memory: in the shadow of `buf`.
  size_t scale, offset;
  __asan_get_shadow_mapping(&scale, &offset);
  char *shadow = (char *)(((size_t)buf >> scale) + offset);
  assert(__asan_region_is_poisoned(shadow, 50) == shadow);
#if defined(__x86_64__)
  // Checked on x86_64 only, where the top of the address space is never
  // application memory. On 32-bit targets it can be.
  assert(__asan_region_is_poisoned((void *)-100, 50) == (void *)-100);
  assert(__asan_region_is_poisoned(buf, (size_t)-1 - (size_t)buf) != nullptr);
#endif
  free(buf);
  return 0;
}
