// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Free-list reuse correctness test for right-align mode.
//
// Both 17 and 25 bytes land in the 32-byte class. When the 17-byte slot is
// freed, Deallocate must push the slot BASE (not the shifted pointer slot_base+15)
// onto the free list. The subsequent 25-byte allocation then reuses the same
// slot but with a different offset (slot_base+7), and all 25 bytes must be
// accessible without OOB.

#include <cstdio>
#include <cstdlib>

int main() {
  // First allocation: 17 bytes → offset 15 within 32-byte slot.
  char *a = (char *)malloc(17);
  if (!a) return 1;
  for (int i = 0; i < 17; i++) a[i] = (char)i;
  free(a);

  // Second allocation: 25 bytes → offset 7 within the same (reused) 32-byte slot.
  char *b = (char *)malloc(25);
  if (!b) return 1;

  // Write and read back all 25 bytes — none must trigger OOB.
  for (int i = 0; i < 25; i++) b[i] = (char)(i + 1);
  for (int i = 0; i < 25; i++)
    if (b[i] != (char)(i + 1)) return 2;

  free(b);

  // CHECK: free_list_reuse: ok
  // CHECK-NOT: LOWFAT ERROR
  printf("free_list_reuse: ok\n");
  return 0;
}
