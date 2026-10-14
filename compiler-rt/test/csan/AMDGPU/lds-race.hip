// RUN: %clang_hip_csan %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#include "race.h"

__global__ void kernel() {
  __shared__ volatile int Shared[64];
  RACE_UNTIL_FOUND(i)
    Shared[0] += i;
}

int main() {
  kernel<<<64, 64>>>();
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: {{.*}}race
// CHECK: {{Write|Read}} of size 4 at 0x{{[1-9a-f][0-9a-f]+}}
// CHECK: #0 {{.*}}lds-race.hip:{{[0-9]+}}
// CHECK-NOT: Location is global
