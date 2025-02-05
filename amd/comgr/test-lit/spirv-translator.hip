// XFAIL: *
// REQUIRES: comgr-has-spirv
// COM: Generate a SPIRV file from a HIP kernel
// RUN: clang -x hip --offload-arch=amdgcnspirv -nogpulib -nogpuinc \
// RUN:    --no-gpu-bundle-output --offload-device-only -O3 %s -o %t.spv

// COM: Run Comgr Translator to covert SPIRV back to LLVM IR
// RUN: spirv-translator %t.spv -o %t.translated.bc

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: llvm-dis %t.translated.bc -o - | FileCheck %s

// COM: Verify LLVM IR text
// CHECK: target triple = "amdgcn-amd-amdhsa"
// CHECK: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z9add_valuePfS_S_

#include <cstdlib>

#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

#ifdef __HIP__
typedef struct hipStream *hipStream_t;
typedef enum hipError {} hipError_t;
int hipConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0,
                     hipStream_t stream = 0);
extern "C" hipError_t __hipPushCallConfiguration(dim3 gridSize, dim3 blockSize,
                                                 size_t sharedSize = 0,
                                                 hipStream_t stream = 0);
extern "C" hipError_t hipLaunchKernel(const void *func, dim3 gridDim,
                                      dim3 blockDim, void **args,
                                      size_t sharedMem,
                                      hipStream_t stream);
#endif

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}
