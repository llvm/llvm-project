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

typedef struct cudaStream *cudaStream_t;

int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0,
                      cudaStream_t stream = 0);

// Host- and device-side placement new overloads.
void *operator new(__SIZE_TYPE__, void *p) { return p; }
void *operator new[](__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new(__SIZE_TYPE__, void *p) { return p; }
__device__ void *operator new[](__SIZE_TYPE__, void *p) { return p; }

#endif // !__NVCC__
