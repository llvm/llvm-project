// Explicit-collect idempotency: a program that collects after a launch and
// then exits normally must not double-count the device counters. The HSA pass
// drains the resident image once, so the merged profile reflects one launch
// (function count 64, even-branch 32). Complements device-early-collect, which
// calls the collector before the first launch.
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
// An explicit collection followed by normal exit records one launch.
// CHECK: _Z8classifyPii:
// CHECK: Function count: 64
// CHECK: Block counts: [0, 32]

#include <hip/hip_runtime.h>
#include <cstdio>

extern "C" int __llvm_profile_hip_collect_device_data(void);

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

  // Explicit mid-program collect after the launch. The atexit drain runs again
  // at exit; together they must not double the device counters.
  (void)__llvm_profile_hip_collect_device_data();

  int h[N];
  (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);

  int evens = 0;
  for (int i = 0; i < N; ++i)
    evens += h[i];
  printf("%s\n", evens == 32 ? "PASS" : "FAIL");
  return evens == 32 ? 0 : 1;
}
