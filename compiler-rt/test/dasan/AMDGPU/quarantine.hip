// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: not %run %t held 2>&1 | FileCheck %s
// RUN: %env_dasan_opts=quarantine_size_mb=0 %run %t reused
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: not %run %t held 2>&1 | FileCheck %s
// RUN: %env_dasan_opts=quarantine_size_mb=0 %run %t reused

__global__ void write_one(int *P) { P[0] = 1; }

int main(int argc, char **argv) {
  const bool Reuse = argv[1][0] == 'r';
  void *A = nullptr, *B = nullptr;
  CHECK_HIP(hipMalloc(&A, 256));
  CHECK_HIP(hipFree(A));
  CHECK_HIP(hipMalloc(&B, 256));
  if ((A == B) != Reuse)
    return 1;

  write_one<<<1, 1>>>((int *)A);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: ERROR: DeviceAddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]]
// CHECK: #0 {{0x[0-9a-f]+}} in write_one{{.*}}quarantine.hip
// CHECK: freed by thread T{{[0-9]+}} here:
// CHECK: quarantine.hip
// CHECK: previously allocated by thread T{{[0-9]+}} here:
// CHECK: quarantine.hip
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-use-after-free
