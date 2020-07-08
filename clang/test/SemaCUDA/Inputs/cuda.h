/* Minimal declarations for CUDA support.  Testing purposes only. */

#include <stddef.h>

// Make this file work with nvcc, for testing compatibility.

#ifndef __NVCC__
#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
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
#else
typedef struct cudaStream *cudaStream_t;
typedef enum cudaError {} cudaError_t;

extern "C" int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                                 size_t sharedSize = 0,
                                 cudaStream_t stream = 0);
extern "C" int __cudaPushCallConfiguration(dim3 gridSize, dim3 blockSize,
                                           size_t sharedSize = 0,
                                           cudaStream_t stream = 0);
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);
#endif

// Host- and device-side placement new overloads.
void *operator new(__SIZE_TYPE__, void *p) { return p; }
void *operator new[](__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new(__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new[](__SIZE_TYPE__, void *p) { return p; }

#endif // !__NVCC__
