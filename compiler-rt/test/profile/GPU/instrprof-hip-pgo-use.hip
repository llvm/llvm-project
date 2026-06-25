// Test the full PGO cycle: instrument, collect, merge, optimize.
// Verifies that the optimized binary produces correct output and that
// profile data is consumed without errors.
//
// REQUIRES: hip, amdgpu
//
// Step 1: Build instrumented binary.
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t.instr \
// RUN:   -L%hip_lib_path -lamdhip64
//
// Step 2: Run to collect profile data.
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: env LLVM_PROFILE_FILE=%t.dir/prof.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.instr 2>&1 | FileCheck %s
//
// Step 3: Merge profile data.
// RUN: llvm-profdata merge -o %t.profdata %t.dir/
//
// Step 4: Build optimized binary with profile data.
// RUN: %clang -x hip -fprofile-instr-use=%t.profdata \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t.opt \
// RUN:   -L%hip_lib_path -lamdhip64 -O2
//
// Step 5: Run optimized binary.
// RUN: env LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.opt 2>&1 | FileCheck %s
//
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void scale(float *data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= factor;
}

int main() {
    constexpr int N = 128;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = (float)i;

    float *d;
    (void)hipMalloc(&d, N * sizeof(float));
    (void)hipMemcpy(d, h, N * sizeof(float), hipMemcpyHostToDevice);

    scale<<<1, N>>>(d, 2.0f, N);

    (void)hipMemcpy(h, d, N * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(d);

    int ok = 1;
    for (int i = 0; i < N; ++i) {
        float expected = (float)(i * 2);
        if (h[i] != expected) ok = 0;
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return !ok;
}
