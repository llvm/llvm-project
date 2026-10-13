// REQUIRES: ubsan-hip
// RUN: %clang_hip -fsanitize=builtin %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clang_hip -O2 -fsanitize=builtin %s -o %t %hip_libs
// RUN: %run %t 2>&1 | FileCheck %s

#define TID __builtin_amdgcn_workitem_id_x()

__device__ int reduce(int V) {
  for (int D = warpSize / 2; D > 0; D /= 2)
    V += __shfl_down(V, D);
  return V;
}

// CHECK-NOT: runtime error
__global__ void clean(int *Out) {
  int Lane = __lane_id();
  int X = Out[TID];
  int V = __builtin_amdgcn_readlane(X, 5);
  V += __shfl(X, Lane + 1) + __shfl_up(X, 1) + __shfl_xor(X, 1) + reduce(X);
  if (Lane < 16)
    V += __shfl(X, Lane ^ 1);
  Out[TID] = V;
}

__global__ void readlane(int *Out) {
  int X = Out[TID];
  // CHECK: lanes.hip:[[#@LINE+1]]:14: runtime error: lane index passed to __builtin_amdgcn_readlane() is not uniform across active lanes
  Out[TID] = __builtin_amdgcn_readlane(X, TID % 4);
}

// The reported location is the builtin inside the hip.h wrapper.
__global__ void shuffle(int *Out) {
  int Lane = __lane_id();
  int X = Out[TID];
  // CHECK: hip.h:[[#]]:[[#]]: runtime error: lane index passed to __builtin_amdgcn_wave_shuffle() refers to an inactive lane
  if (Lane < 16)
    Out[TID] = __shfl(X, Lane + 16);
}

__global__ void shuffle_direct(int *Out) {
  int Lane = __lane_id();
  int X = Out[TID];
  // CHECK: lanes.hip:[[#@LINE+2]]:16: runtime error: lane index passed to __builtin_amdgcn_wave_shuffle() refers to an inactive lane
  if (Lane < 16)
    Out[TID] = __builtin_amdgcn_wave_shuffle(X, Lane + 16);
}

// CHECK: hip.h:[[#]]:[[#]]: runtime error: lane index passed to __builtin_amdgcn_wave_shuffle() refers to an inactive lane
__global__ void partial_reduce(int *Out) { Out[TID] = reduce(Out[TID]); }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 64 * sizeof(int)));
  clean<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  readlane<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  shuffle<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  shuffle_direct<<<1, 64>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  partial_reduce<<<1, 20>>>(P);
  CHECK_HIP(hipDeviceSynchronize());
  hipFree(P);
  return 0;
}
