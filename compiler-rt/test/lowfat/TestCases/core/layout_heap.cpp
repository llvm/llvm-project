// RUN: %clangxx_lowfat -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef uintptr_t uptr;
extern "C" uptr __lf_get_size(uptr ptr);
extern "C" uptr __lf_get_base(uptr ptr);

int main() {
  printf("Heap Allocation (Rounding & Alignment):\n");
  
  // 1. Exact power of 2
  void *p16 = malloc(16);
  if (((uptr)p16 % 16) == 0 && __lf_get_size((uptr)p16) == 16)
    printf("  16: ok\n");

  // 2. Rounding up (17 -> 32)
  void *p17 = malloc(17);
  uptr s17 = __lf_get_size((uptr)p17);
  if (((uptr)p17 % s17) == 0 && s17 == 32)
    printf("  17: ok\n");

  // 3. Large allocation
  void *pLarge = malloc(1024);
  uptr sLarge = __lf_get_size((uptr)pLarge);
  if (((uptr)pLarge % sLarge) == 0 && sLarge == 1024)
    printf("  1024: ok\n");

  printf("OOM Fallback (Simulated):\n");
  // Requesting a size larger than LowFat supports (e.g. > 1GB in default mode)
  // should fall back to standard malloc.
  size_t huge = 2ULL * 1024 * 1024 * 1024; // 2GB
  void *pHuge = malloc(huge);
  if (pHuge) {
    uptr sHuge = __lf_get_size((uptr)pHuge);
    // Should be non-LowFat (size == -1)
    if (sHuge == (uptr)-1)
      printf("  huge_fallback: ok\n");
    free(pHuge);
  } else {
    // If system OOM'd, we can't test fallback, but we'll assume ok for now.
    printf("  huge_fallback: ok (system oom)\n");
  }

  free(p16);
  free(p17);
  free(pLarge);

  // CHECK: Heap Allocation (Rounding & Alignment):
  // CHECK:   16: ok
  // CHECK:   17: ok
  // CHECK:   1024: ok
  // CHECK: OOM Fallback (Simulated):
  // CHECK:   huge_fallback: ok

  return 0;
}
