// RUN: %clang_hwasan %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/hwasan_interface.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  __hwasan_enable_allocator_tagging();
  for (int sz = 0; sz < 64; ++sz) {
    fprintf(stderr, "sz: %d\n", sz);
    char *x = (char *)malloc(sz);
    do {
      // Empty range is always OK.
      for (int b = -16; b < sz + 32; ++b)
        assert(__hwasan_test_shadow(x + b, 0) == -1);

      int real_sz = sz ? sz : 1;
      // Unlucky case when we cant distinguish between tag and short granule size.
      if (__hwasan_tag_pointer(x, real_sz % 16) == x)
        break;

      // Underflow - the first byte is bad.
      for (int b = -16; b < 0; ++b)
        assert(__hwasan_test_shadow(x + b, real_sz) == 0);

      // Inbound ranges.
      for (int b = 0; b < real_sz; ++b)
        for (int e = b; e <= real_sz; ++e)
          assert(__hwasan_test_shadow(x + b, e - b) == -1);

      // Overflow - the first byte after the buffer is bad.
      for (int b = 0; b <= real_sz; ++b)
        for (int e = real_sz + 1; e <= real_sz + 64; ++e)
          assert(__hwasan_test_shadow(x + b, e - b) == (real_sz - b));

    } while (0);
    free(x);
  }
  return 0;
}
