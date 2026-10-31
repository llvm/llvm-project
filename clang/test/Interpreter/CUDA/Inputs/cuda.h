/* Minimal declarations for CUDA support.  Testing purposes only. */

#include <stddef.h>

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)
      : x(x), y(y), z(z) {}
};

typedef struct cudaStream *cudaStream_t;
typedef enum cudaError {} cudaError_t;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};

extern "C" int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                                 size_t sharedSize = 0,
                                 cudaStream_t stream = 0);
extern "C" __device__ cudaError_t cudaLaunchDevice(void *func,
                                                   void *parameterBuffer,
                                                   dim3 gridDim, dim3 blockDim,
                                                   unsigned int sharedMem,
                                                   cudaStream_t stream);
extern "C" __device__ void *cudaGetParameterBuffer(size_t alignment,
                                                   size_t size);

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size);
extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  cudaMemcpyKind kind);
extern "C" cudaError_t cudaGetLastError();
