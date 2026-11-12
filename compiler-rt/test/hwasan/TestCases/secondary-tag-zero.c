// Test that tagging of freed large region is 0, which is better for RSS usage.
// RUN: %clang_hwasan -mllvm -hwasan-globals=0 -mllvm -hwasan-instrument-stack=0 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <stdlib.h>

#include <sanitizer/hwasan_interface.h>

const int kSize = 10000000;

void *p;
int main() {
  for (int i = 0; i < 256; ++i) {
    p = malloc(kSize);
    assert(-1 == __hwasan_test_shadow(p, kSize));

    free(p);
    assert(-1 == __hwasan_test_shadow(__hwasan_tag_pointer(p, 0), kSize));
  }
  return 0;
}
