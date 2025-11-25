// RUN: %ROCM_ENV && %hipcompiler -O0 -ggdb --offload-arch=gfx908:xnack+ %s -o %t && not %run %t 10 1 11 10 2>&1 | FileCheck %s
// CHECK: AddressSanitizer: heap-buffer-overflow on amdgpu device
// CHECK-NEXT: {{WRITE of size 4 in workgroup id}}
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

__global__ void
set1(int *p)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    p[i] = 77;
}

extern "C"
__attribute__((no_sanitize_address))
const char* __asan_default_options() { return "detect_leaks=0"; }

int
main(int argc, char **argv)
{
    int m  = std::atoi(argv[1]);
    int n1 = std::atoi(argv[2]);
    int n2 = std::atoi(argv[3]);
    int c  = std::atoi(argv[4]);
    int *dp;
    hipMalloc(&dp, m*sizeof(int));
    hipLaunchKernelGGL(set1, dim3(n1), dim3(n2), 0, 0, dp);
    int *hp = (int*)malloc(c*sizeof(int));
    hipMemcpy(hp, dp, m*sizeof(int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    hipFree(dp);
    free(hp);
    return 0;
}
