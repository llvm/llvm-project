// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not="runtime error"

// TODO: Device __int128 operands are passed by pointer and dropped on the host.
__global__ void overflow(__int128 *Out) {
  __int128 A = (__int128)1 << 126;
  *Out = A + A;
}

int main() {
  __int128 *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(__int128)));
  overflow<<<1, 1>>>(P);
  hipDeviceSynchronize();
  printf("survived\n");
  hipFree(P);
  return 0;
}

// CHECK: survived
