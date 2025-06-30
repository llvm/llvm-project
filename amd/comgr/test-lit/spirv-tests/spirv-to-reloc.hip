// REQUIRES: comgr-has-spirv

// COM: Generate a SPIR-V file from a HIP kernel
// RUN: clang -x hip --offload-arch=amdgcnspirv -nogpulib -nogpuinc \
// RUN:    --no-gpu-bundle-output --offload-device-only -O3 %s -o %t.spv \
// RUN:    -fvisibility=hidden -fno-autolink -fexceptions -fcolor-diagnostics

// COM: Compile SPIR-V source to a relocatable
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=spirv-flags.txt \
// RUN:   spirv-to-reloc %t.spv %t.o

// COM: Check that SPIR-V flags are correctly extracted
// RUN: grep '\-fvisibility=hidden' spirv-flags.txt
// RUN: grep '\-fno-autolink' spirv-flags.txt
// RUN: grep '\-fexceptions' spirv-flags.txt
// RUN: grep '\-fcolor-diagnostics' spirv-flags.txt
// RUN: grep '\-O3' spirv-flags.txt
// RUN: grep '\-mcode-object-version=5' spirv-flags.txt

// RUN: rm spirv-flags.txt

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
