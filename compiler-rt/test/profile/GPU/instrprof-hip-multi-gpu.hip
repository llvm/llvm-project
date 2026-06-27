// Test that HIP PGO works on multi-GPU systems. The kernel runs on the default
// device, so the host-shadow drain (guarded by upstream's launch tracking)
// collects only that device and the supplemental HSA agent-walk then finds the
// same code object and dedups it out. The point of the test is that neither
// pass crashes or hangs reading a non-resident device on a host with several
// GPUs (the failure mode that the launch tracking + HSA residency walk fix).
//
// REQUIRES: hip, amdgpu
// The "walk complete" / dedup notes are Linux-only HSA-drain strings; the
// Windows host-shadow drain collects only the current device.
// UNSUPPORTED: windows
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// The host-shadow pass drains the launched device, the HSA walk finds that same
// code object and dedups it (drained=0), and the program does not crash.
// CHECK: Copied device sections:
// CHECK: device bounds already drained, skipping
// CHECK: walk complete: agents={{[0-9]+}} pairs={{[0-9]+}} found={{[1-9][0-9]*}} drained={{[0-9]+}}
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void add_one(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] += 1;
}

int main() {
    int ndev = 0;
    (void)hipGetDeviceCount(&ndev);

    constexpr int N = 32;
    int h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i;

    int *d_data;
    (void)hipMalloc(&d_data, N * sizeof(int));
    (void)hipMemcpy(d_data, h_data, N * sizeof(int), hipMemcpyHostToDevice);

    add_one<<<1, N>>>(d_data, N);

    (void)hipMemcpy(h_data, d_data, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d_data);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h_data[i] != i + 1) ok = 0;

    printf("%s (devices=%d)\n", ok ? "PASS" : "FAIL", ndev);
    return !ok;
}
