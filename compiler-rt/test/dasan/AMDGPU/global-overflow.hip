// REQUIRES: dasan-globals
// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip -Xarch_device -fsanitize=daddress -shared-libsan %s -o %t \
// RUN:   %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__device__ int Table[16];

__global__ void write_past(int I) { Table[I] = 1; }

int main() {
  write_past<<<1, 1>>>(16);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: global-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}global-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after global variable 'Table' {{.*}} of size 64
// CHECK: SUMMARY: DeviceAddressSanitizer: global-buffer-overflow
