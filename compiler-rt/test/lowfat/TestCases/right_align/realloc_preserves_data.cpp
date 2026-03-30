// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Regression test for right-align mode realloc semantics.
//
// A 48-byte allocation lands in a 64-byte slot and, with 16-byte malloc
// alignment, is returned at slot_base+16. realloc() must preserve the 48 bytes
// of user data starting at the returned pointer, not copy from slot_base.

#include <cstdio>
#include <cstdlib>

int main() {
  constexpr int OldSize = 48;
  constexpr int NewSize = 64;

  unsigned char *p = (unsigned char *)malloc(OldSize);
  if (!p) return 1;

  for (int i = 0; i < OldSize; ++i)
    p[i] = (unsigned char)(0xA0 + i);

  unsigned char *q = (unsigned char *)realloc(p, NewSize);
  if (!q) return 2;

  for (int i = 0; i < OldSize; ++i) {
    if (q[i] != (unsigned char)(0xA0 + i)) {
      printf("realloc mismatch at %d: got 0x%02x expected 0x%02x\n",
             i, (unsigned)q[i], (unsigned)(0xA0 + i));
      free(q);
      return 3;
    }
  }

  // CHECK: realloc_preserves_data: ok
  // CHECK-NOT: realloc mismatch
  printf("realloc_preserves_data: ok\n");
  free(q);
  return 0;
}
