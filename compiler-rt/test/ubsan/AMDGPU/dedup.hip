// REQUIRES: ubsan-hip
// RUN: %clang_ubsan_hip %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

__global__ void overflow(int *Out) {
  int A = 0x7fffffff;
  for (int I = 0; I < 8; ++I)
    *Out = A + 1;
}

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, sizeof(int)));
  overflow<<<1, 1>>>(P);
  hipDeviceSynchronize();
  printf("survived\n");
  hipFree(P);
  return 0;
}

// CHECK: runtime error: signed integer overflow
// CHECK-NOT: runtime error:
// CHECK: survived
