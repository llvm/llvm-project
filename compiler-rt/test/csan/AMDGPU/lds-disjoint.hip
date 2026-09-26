// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --allow-empty

#include "race.h"

// Each workgroup has its own LDS. Many groups touching the same offset must
// not look like a race once the key carries the workgroup index.
__global__ void kernel() {
  __shared__ int Shared[64];
  unsigned Slot = CSAN_TID() % 64;
  for (int I = 0; I < 1024; ++I)
    Shared[Slot] += I;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
