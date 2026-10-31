// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

#define N 512

__device__ int Buf[N];

__global__ void kernel() {
  unsigned Id = CSAN_TID();
  RACE_UNTIL_FOUND(i) {
    if (Id == 0)
      __builtin_memmove((void *)Buf, (const void *)(Buf + 1),
                        (N - 1) * sizeof(int));
    else
      Buf[N - 1] = Id;
  }
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}memmove-race.hip:{{[0-9]+}}
