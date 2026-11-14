// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -shared-libsan %s -o %t %hip_libs
// RUN: llvm-readelf -d %t | FileCheck %s --check-prefix=NEEDED
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void write_past(int *P, int I) { P[I] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  write_past<<<1, 1>>>(P, 64);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread (0,0,0) of block (0,0,0)
// CHECK: #0 {{0x[0-9a-f]+}} in write_past{{.*}}heap-overflow.hip
// CHECK: [[ADDR]] is located 0 bytes after 256-byte region
// CHECK: allocated by thread T{{[0-9]+}} here:
// CHECK: heap-overflow.hip
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: HINT: build with -fsanitize-recover=daddress

// NEEDED: libclang_rt.dasan.so
