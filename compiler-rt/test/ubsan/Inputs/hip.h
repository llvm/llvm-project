//===-- hip.h -------------------------------------------------------------===//
//
// Minimal HIP headers so the tests can be compiled without a ROCm installation.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_TEST_HIP_H
#define UBSAN_TEST_HIP_H

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
      : x(x), y(y), z(z) {}
};

extern "C" {
typedef struct ihipStream_t *hipStream_t;

int hipMalloc(void **Ptr, unsigned long Size);
int hipFree(void *Ptr);
int hipDeviceSynchronize(void);

int __hipPushCallConfiguration(dim3 GridDim, dim3 BlockDim,
                               unsigned long SharedMem = 0,
                               hipStream_t Stream = 0);
int __hipPopCallConfiguration(dim3 *GridDim, dim3 *BlockDim,
                              unsigned long *SharedMem, hipStream_t *Stream);
int hipLaunchKernel(const void *Func, dim3 GridDim, dim3 BlockDim, void **Args,
                    unsigned long SharedMem, hipStream_t Stream);

int printf(const char *, ...);
}

#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    if ((Expr) != 0) {                                                         \
      printf("setup failed: %s\n", #Expr);                                     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#endif // UBSAN_TEST_HIP_H
