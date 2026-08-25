// Test HIP coverage mapping produces source-level coverage for host code.
//
// REQUIRES: hip, amdgpu
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: env LLVM_PROFILE_FILE=%t.dir/prof.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.dir/
// RUN: llvm-cov report %t -instr-profile=%t.profdata 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REPORT
//
// REPORT: instrprof-hip-coverage.hip
// No coverage column should be fully uncovered. Anchor on a non-digit before
// the "0.00%" so this does not spuriously match e.g. "80.00%".
// REPORT-NOT: {{[^.0-9]0[.]00%}}

#include <hip/hip_runtime.h>
#include <cstdio>

__device__ int gpu_abs(int x) {
    return x < 0 ? -x : x;
}

__global__ void abs_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = gpu_abs(data[idx]);
}

int main() {
    constexpr int N = 16;
    int h[N];
    for (int i = 0; i < N; ++i)
        h[i] = (i % 2 == 0) ? i : -i;

    int *d;
    (void)hipMalloc(&d, N * sizeof(int));
    (void)hipMemcpy(d, h, N * sizeof(int), hipMemcpyHostToDevice);
    abs_kernel<<<1, N>>>(d, N);
    (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h[i] != i) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return !ok;
}
