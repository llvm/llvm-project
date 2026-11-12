// RUN: %clang_hip_dasan --offload-device-only --no-gpu-bundle-output \
// RUN:   -foffload-lto %s -o %t.elf
// RUN: llvm-readelf --notes %t.elf | FileCheck %s \
// RUN:   --implicit-check-not='group_segment_fixed_size: {{[23][0-9][0-9][0-9][0-9]}}'
// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %run %t

#define N 16

__global__ void k1(float *Out, int I) {
  __shared__ float S[N][N][N];
  float *P = &S[0][0][0];
  P[I] = 1.f;
  *Out = S[0][0][0];
}

__global__ void k2(float *Out, int I) {
  __shared__ float T[N][N][N];
  float *P = &T[0][0][0];
  P[I] = 2.f;
  *Out = T[0][0][0];
}

int main() {
  float *Out = nullptr;
  CHECK_HIP(hipMalloc((void **)&Out, sizeof(float)));
  k1<<<1, 1>>>(Out, 0);
  k2<<<1, 1>>>(Out, 0);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: .group_segment_fixed_size: 16384
// CHECK: .group_segment_fixed_size: 16384
