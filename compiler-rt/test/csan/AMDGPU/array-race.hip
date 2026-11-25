// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__device__ volatile int data[64];

__global__ void kernel() {
  unsigned Slot = CSAN_TID() % 64;
  RACE_UNTIL_FOUND(i)
    data[Slot]++;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}array-race.hip:{{[0-9]+}}
// CHECK: Location is global '{{.*}}data{{.*}}' of size {{[0-9]+}} at 0x{{.*}}
