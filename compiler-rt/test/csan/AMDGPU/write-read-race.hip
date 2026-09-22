// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__device__ volatile int Data;

__global__ void kernel() {
  int Sink = 0;
  RACE_UNTIL_FOUND(i) {
    Data = i;
    Sink += Data;
  }
  if (Sink == -1)
    __builtin_trap();
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: #0 {{.*}}write-read-race.hip:{{[0-9]+}}
// CHECK: Location is global '{{.*}}Data{{.*}}' of size {{[0-9]+}} at 0x{{.*}}
