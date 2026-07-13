// Test basic HIP PGO instrumentation and profile collection.
//
// REQUIRES: hip, amdgpu
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: env LLVM_PROFILE_FILE=%t.dir/prof.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t
// RUN: ls %t.dir/prof.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.dir/
// RUN: llvm-profdata show --all-functions %t.profdata \
// RUN:   | FileCheck %s --check-prefix=PROF
//
// PROF: _Z6squarePiPKii
// PROF: main
// PROF: Functions shown: 2
// PROF: Total functions: 2

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void square(int *out, const int *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = in[idx] * in[idx];
}

int main() {
    constexpr int N = 64;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = i;

    int *d_in, *d_out;
    (void)hipMalloc(&d_in, N * sizeof(int));
    (void)hipMalloc(&d_out, N * sizeof(int));
    (void)hipMemcpy(d_in, h_in, N * sizeof(int), hipMemcpyHostToDevice);

    square<<<1, N>>>(d_out, d_in, N);

    (void)hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h_out[i] != i * i) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    (void)hipFree(d_in);
    (void)hipFree(d_out);
    return !ok;
}
