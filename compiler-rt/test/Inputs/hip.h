//===-- hip.h -------------------------------------------------------------===//
//
// Minimal HIP headers so the tests can be compiled without a ROCm installation.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_RT_TEST_HIP_H
#define COMPILER_RT_TEST_HIP_H

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

__device__ inline int __hip_warp_size() {
  return __builtin_amdgcn_wavefrontsize();
}
#define warpSize __hip_warp_size()

__device__ inline int __lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0));
}

__device__ inline int __shfl(int Var, int SrcLane, int Width = warpSize) {
  int Self = __lane_id();
  int Index = (SrcLane & (Width - 1)) + (Self & ~(Width - 1));
  return __builtin_amdgcn_wave_shuffle(Var, Index);
}

__device__ inline int __shfl_up(int Var, unsigned Delta, int Width = warpSize) {
  int Self = __lane_id();
  int Index = Self - Delta;
  Index = Index < (Self & ~(Width - 1)) ? Self : Index;
  return __builtin_amdgcn_wave_shuffle(Var, Index);
}

__device__ inline int __shfl_down(int Var, unsigned Delta,
                                  int Width = warpSize) {
  int Self = __lane_id();
  int Index =
      (int)((Self & (Width - 1)) + Delta) >= Width ? Self : Self + Delta;
  return __builtin_amdgcn_wave_shuffle(Var, Index);
}

__device__ inline int __shfl_xor(int Var, int LaneMask, int Width = warpSize) {
  int Self = __lane_id();
  int Index = Self ^ LaneMask;
  Index = Index >= ((Self + Width) & ~(Width - 1)) ? Self : Index;
  return __builtin_amdgcn_wave_shuffle(Var, Index);
}

#define CHECK_HIP(Expr)                                                        \
  do {                                                                         \
    if ((Expr) != 0) {                                                         \
      printf("setup failed: %s\n", #Expr);                                     \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#endif // COMPILER_RT_TEST_HIP_H
