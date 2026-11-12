// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  P[0] = 1;
  if (P[0] != 1)
    return 1;
  P[64] = 1;
  fprintf(stderr, "missed\n");
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in main {{.*}}host-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: allocated by thread T{{[0-9]+}} here:
// CHECK: host-overflow.hip
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: HINT: build with -fsanitize-recover=daddress
