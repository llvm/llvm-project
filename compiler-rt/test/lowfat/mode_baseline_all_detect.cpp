// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OOB
// RUN: %clangxx_lowfat_safe -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OOB
// RUN: %clangxx_lowfat -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OOB
// RUN: %clangxx_lowfat_safe -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-OOB

// Baseline: all three modes detect an OOB access whose result IS observable
// Contrast with mode_diff_pure_call.cpp where a discarded-result call lets
// Fast eliminate the load before the LowFat pass even sees it.

#include <cstdio>
#include <cstdlib>

volatile char sink; // volatile global: any write/read here is always observable

int main() {
  char *p = (char *)malloc(16);

  // 8-byte (double) OOB read at offset 14: bytes [14, 22) overflow the
  // 16-byte LowFat slot [0, 16). 
  sink = (char)(*reinterpret_cast<volatile double *>(p + 14));

  free(p);

  // CHECK-OOB: LOWFAT ERROR: out-of-bounds error detected!
  printf("DONE\n");
  return 0;
}
