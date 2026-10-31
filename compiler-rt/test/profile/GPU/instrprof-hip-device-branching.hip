// Test that device-side branching is captured in profile counters.
// Exercises the classify-style pattern where different branches are taken
// by different threads, verifying that counter values reflect actual execution.
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
// Device functions should appear with non-zero counters. The __device__
// classify() helper is inlined into the histogram kernel, so it does not get a
// separate profile record; its branching is captured within the kernel's
// counters instead.
// PROF-DAG: _Z9histogramPKiPii
// PROF-DAG: main
// PROF: Total functions: 2
// PROF: Maximum function count: {{[1-9][0-9]*}}

#include <hip/hip_runtime.h>
#include <cstdio>

__device__ int classify(int x) {
    if (x > 100)    return 2;
    else if (x > 0) return 1;
    else            return 0;
}

__global__ void histogram(const int *input, int *bins, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cls = classify(input[idx]);
        atomicAdd(&bins[cls], 1);
    }
}

int main() {
    constexpr int N = 256;
    constexpr int NBINS = 3;

    int h_in[N], h_bins[NBINS] = {};
    for (int i = 0; i < N; ++i)
        h_in[i] = (i % 3 == 0) ? -1 : (i % 3 == 1) ? 50 : 200;

    int *d_in, *d_bins;
    (void)hipMalloc(&d_in, N * sizeof(int));
    (void)hipMalloc(&d_bins, NBINS * sizeof(int));
    (void)hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice);
    (void)hipMemset(d_bins, 0, NBINS * sizeof(int));

    histogram<<<1, N>>>(d_in, d_bins, N);

    (void)hipMemcpy(h_bins, d_bins, NBINS * sizeof(int), hipMemcpyDeviceToHost);
    printf("bins: [%d, %d, %d]\n", h_bins[0], h_bins[1], h_bins[2]);

    int ok = (h_bins[0] > 0 && h_bins[1] > 0 && h_bins[2] > 0);
    printf("%s\n", ok ? "PASS" : "FAIL");

    (void)hipFree(d_in);
    (void)hipFree(d_bins);
    return !ok;
}
