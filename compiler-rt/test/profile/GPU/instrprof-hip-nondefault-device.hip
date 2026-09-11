// Test PGO when the kernel runs on a non-default device (here the program
// selects device 1). Upstream's launch tracking records that device 1 was used,
// so the host-shadow drain skips the other devices and collects device 1, and
// the supplemental HSA agent-walk then finds that same code object and dedups it
// out. This exercises both that the correct device is drained and that an
// unused device is never read (which would fault/hang on a multi-GPU host).
//
// REQUIRES: hip, amdgpu, multi-device
// The "walk complete" / dedup notes are Linux-only HSA-drain strings; the
// Windows host-shadow drain only collects the current device.
// UNSUPPORTED: windows
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 %run %t 2>&1 | FileCheck %s
//
// The launched device (1) is drained, the unused default device is skipped, and
// the HSA walk finds the same code object and dedups it (drained=0).
// CHECK: Skipping unused device 0
// CHECK: Collecting static profile data from device 1
// CHECK: Copied device sections:
// CHECK: device bounds already drained, skipping
// CHECK: walk complete: agents={{[0-9]+}} pairs={{[0-9]+}} found={{[1-9][0-9]*}} drained={{[0-9]+}}
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void fill(int *data, int val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = val;
}

int main() {
    int ndev = 0;
    (void)hipGetDeviceCount(&ndev);
    if (ndev < 2) {
        printf("PASS (skipped: only %d device)\n", ndev);
        return 0;
    }

    (void)hipSetDevice(1);

    constexpr int N = 32;
    int h[N] = {};
    int *d;
    (void)hipMalloc(&d, N * sizeof(int));
    fill<<<1, N>>>(d, 99, N);
    (void)hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(d);

    int ok = 1;
    for (int i = 0; i < N; ++i)
        if (h[i] != 99) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return !ok;
}
