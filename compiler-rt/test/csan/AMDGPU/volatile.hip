// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %clang_hip_csan -mllvm -csan-distinguish-volatile %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=MARKED --allow-empty

#include "race.h"

__device__ volatile int Data;

__global__ void kernel() {
  for (int I = 0; I < 1 << 14 && !race_found(); ++I)
    Data = I;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// RACE: WARNING: ConcurrencySanitizer: {{.*}}race
// MARKED-NOT: WARNING: ConcurrencySanitizer
