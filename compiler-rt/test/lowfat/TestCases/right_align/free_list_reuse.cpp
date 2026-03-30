// RUN: %clangxx_lowfat_right_align -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_right_align -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Free-list reuse correctness test for right-align mode.
//
// Both 48 and 63 bytes land in the 64-byte class. When the 48-byte slot is
// freed, Deallocate must push the slot BASE (not the shifted pointer
// slot_base+16) onto the free list. The subsequent 63-byte allocation then
// reuses the same slot with a different aligned offset, and all 63 bytes must
// be accessible without OOB.

#include <cstdio>
#include <cstdlib>

int main() {
  // First allocation: 48 bytes -> offset 16 within a 64-byte slot.
  char *a = (char *)malloc(48);
  if (!a) return 1;
  for (int i = 0; i < 48; i++) a[i] = (char)i;
  free(a);

  // Second allocation: 63 bytes -> offset 0 within the same reused 64-byte slot.
  char *b = (char *)malloc(63);
  if (!b) return 1;

  // Write and read back all 63 bytes — none must trigger OOB.
  for (int i = 0; i < 63; i++) b[i] = (char)(i + 1);
  for (int i = 0; i < 63; i++)
    if (b[i] != (char)(i + 1)) return 2;

  free(b);

  // CHECK: free_list_reuse: ok
  // CHECK-NOT: LOWFAT ERROR
  printf("free_list_reuse: ok\n");
  return 0;
}
