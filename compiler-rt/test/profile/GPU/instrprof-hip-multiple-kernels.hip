// Test PGO with multiple kernel launches from a single TU.
// Verifies that counters from all device functions are collected correctly.
//
// REQUIRES: hip, amdgpu
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: env LLVM_PROFILE_FILE=%t.dir/prof.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.dir/
// RUN: llvm-profdata show --all-functions %t.profdata \
// RUN:   | FileCheck %s --check-prefix=PROF
//
// All three kernels plus main should be profiled.
// PROF-DAG: _Z4fillPii
// PROF-DAG: _Z5scalePii
// PROF-DAG: _Z6negatePii
// PROF-DAG: main
// PROF: Total functions: 4

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void fill(int *data, int val) {
    data[threadIdx.x] = val;
}

__global__ void scale(int *data, int factor) {
    data[threadIdx.x] *= factor;
}

__global__ void negate(int *data, int n) {
    int idx = threadIdx.x;
    if (idx < n)
        data[idx] = -data[idx];
}

int main() {
    constexpr int N = 16;
    int h[N];
    int *d;
    (void)hipMalloc(&d, N * sizeof(int));

    fill<<<1, N>>>(d, 5);
    scale<<<1, N>>>(d, 3);
    negate<<<1, N>>>(d, N);

    (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h[i] != -15) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return !ok;
}
