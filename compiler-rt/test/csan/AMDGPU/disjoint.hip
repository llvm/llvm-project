// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --allow-empty

#include "race.h"

__device__ int data[64 * 64];

__global__ void kernel() {
  unsigned Id = CSAN_TID();
  for (int I = 0; I < 1024; ++I)
    data[Id % (64 * 64)] += I;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
