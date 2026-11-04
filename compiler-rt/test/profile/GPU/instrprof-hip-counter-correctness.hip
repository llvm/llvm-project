// Quantitative device-counter correctness: the drained device profile must carry
// the *exact* per-region execution counts produced by the kernel, not merely
// "some counts are present". A single launch of 64 threads over classify() must
// record a function entry count of 64, an even-branch block count of 32, and
// matching block uniformity data. This pins the drain + dedup path against
// silent under/over-counting (e.g. a dedup bug that dropped or doubled a section
// would change these numbers).
//
// REQUIRES: hip, amdgpu
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/a.out
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/a.profdata
// RUN: llvm-profdata show --all-functions --counts %t.dir/a.profdata \
// RUN:   | FileCheck %s
//
// The device kernel ran with exactly 64 threads, all of which entered the
// function; the even branch was taken 32 times and the early-return path 0.
// CHECK: _Z8classifyPii:
// CHECK: Function count: 64
// CHECK: Block counts: [0, 32]
// CHECK-NEXT: Block uniformity: [U, D]

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
