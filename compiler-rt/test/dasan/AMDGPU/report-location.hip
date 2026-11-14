// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -O2 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip_dasan -fsanitize-recover=daddress -O2 %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

__device__ inline __attribute__((always_inline)) void put(int *P, long I) {
  P[I] = 0; // CHECK-DAG: #0 {{.*}} in put{{.*}}report-location.hip:[[@LINE]]:
}

#define AT(K) (W == K ? N : 0)

__global__ void sites(int *A, int *B, int *C, long W, long N, long Trip) {
  put(A, AT(0));    // CHECK-DAG: #1 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  A[AT(1)] = 1;     // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  B[AT(2)] = 2;     // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  int X = A[AT(3)]; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  int Y = B[AT(4)]; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  A[AT(5) + 1] = X; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  if (threadIdxX() == 0) {
    A[AT(6) + 2] = 6; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
    B[AT(7) + 2] = 7; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  }
#pragma clang loop unroll(disable)
  for (long I = 0; I < Trip; ++I)
    C[I * AT(8)] = I;   // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
  C[AT(9) + 1] = X + Y; // CHECK-DAG: #0 {{.*}} in sites{{.*}}.hip:[[@LINE]]:
}

// Identical checks on sibling paths are kept apart only by nomerge. The sleeps
// stop the stores from being sunk into one before instrumentation.
__global__ void twins(int *P, long I, long W) {
  if (W == 0) {
    P[I] = 0; // CHECK-DAG: #0 {{.*}} in twins{{.*}}.hip:[[@LINE]]:
    __builtin_amdgcn_s_sleep(1);
  } else {
    P[I] = 1; // CHECK-DAG: #0 {{.*}} in twins{{.*}}.hip:[[@LINE]]:
    __builtin_amdgcn_s_sleep(2);
  }
}

int main() {
  const long N = 60;
  int *A = nullptr, *B = nullptr, *C = nullptr;
  CHECK_HIP(hipMalloc((void **)&A, N * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&B, N * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&C, N * sizeof(int)));

  for (long W = 0; W < 10; ++W)
    sites<<<1, 1>>>(A, B, C, W, N, 2);
  for (long W = 0; W < 2; ++W)
    twins<<<1, 1>>>(C, N, W);
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}
