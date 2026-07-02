// Multi-process accumulation: device counters from several independent runs of an
// instrumented HIP program must accumulate when their profraws are offline-merged
// (llvm-profdata merge), the common real-world feedback-collection path. Three
// runs of a 64-thread launch must sum to a function count of 192 and an
// even-branch block count of 96 (3 x 64 / 3 x 32).
//
// Note: on-the-fly merge-pooling via LLVM_PROFILE_FILE=...%m does NOT currently
// accumulate *device* counters -- the device profraw is rewritten per process
// rather than merged in place -- so each process must write a distinct file
// (here via %p) and the accumulation is done by llvm-profdata merge.
//
// REQUIRES: hip, amdgpu
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/run1.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/a.out
// RUN: env LLVM_PROFILE_FILE=%t.dir/run2.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/a.out
// RUN: env LLVM_PROFILE_FILE=%t.dir/run3.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/a.out
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/a.profdata
// RUN: llvm-profdata show --all-functions --counts %t.dir/a.profdata \
// RUN:   | FileCheck %s
//
// CHECK: _Z8classifyPii:
// CHECK: Function count: 192
// CHECK: Block counts: [0, 96]

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void classify(int *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  if (idx % 2 == 0)
    out[idx] = 1;
  else
    out[idx] = 0;
}

int main() {
  constexpr int N = 64;
  int *d = nullptr;
  if (hipMalloc(&d, N * sizeof(int)) != hipSuccess)
    return 2;
  classify<<<1, N>>>(d, N);
  (void)hipDeviceSynchronize();
  int h[N];
  (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);

  int evens = 0;
  for (int i = 0; i < N; ++i)
    evens += h[i];
  printf("%s\n", evens == 32 ? "PASS" : "FAIL");
  return evens == 32 ? 0 : 1;
}
