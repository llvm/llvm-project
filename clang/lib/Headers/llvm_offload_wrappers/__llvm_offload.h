/*===------ LLVM/Offload helpers for kernel languages (CUDA/HIP) -*- c++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#include <stddef.h>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

#if defined(__NVPTX__) || defined(__AMDGPU__) || defined(__SPIRV__)
#define __LLVM_OFFLOAD_HAS_GPU_INTRINSICS 1
#pragma push_macro("_OPENMP")
#define _OPENMP
#include <gpuintrin.h>
#pragma pop_macro("_OPENMP")
#else
#define __LLVM_OFFLOAD_HAS_GPU_INTRINSICS 0
#endif

#if defined(__CUDA__) || defined(__HIP__) || __LLVM_OFFLOAD_HAS_GPU_INTRINSICS
#define __LLVM_OFFLOAD_DEVICE_ATTR __attribute__((device))
#define __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __attribute__((device, weak))
#else
#define __LLVM_OFFLOAD_DEVICE_ATTR
#define __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __attribute__((weak))
#endif

extern "C" {

typedef struct dim3 {
  dim3() {}
  dim3(unsigned x) : x(x) {}
  unsigned x = 0, y = 0, z = 0;
} dim3;

// TODO: For some reason the CUDA device compilation requires this declaration
// to be present on the device while it is only used on the host.
unsigned __llvmPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0, void *stream = 0);
}

__LLVM_OFFLOAD_DEVICE_ATTR inline void __syncthreads(void) {
#if __LLVM_OFFLOAD_HAS_GPU_INTRINSICS
  __gpu_sync_threads();
#endif
}

// Make sure nobody can create instances of the coordinate types, take their
// address, copy, or assign them.
#pragma push_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")
#define __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag)                                \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag() = delete;                                 \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag(const __tag &) = delete;                    \
  __LLVM_OFFLOAD_DEVICE_ATTR void operator=(const __tag &) const = delete;     \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag *operator&() const = delete

#pragma push_macro("__GPU_COORD_BUILTIN")
#if __LLVM_OFFLOAD_HAS_GPU_INTRINSICS
#pragma push_macro("__GPU_COORD_GETTER")
#define __GPU_COORD_GETTER(__expr) { return __expr; }

#define __GPU_COORD_BUILTIN(__tag, __fx, __fy, __fz)                           \
  struct __tag {                                                               \
    __declspec(property(get = __get_x)) unsigned int x;                        \
    __declspec(property(get = __get_y)) unsigned int y;                        \
    __declspec(property(get = __get_z)) unsigned int z;                        \
    static inline __LLVM_OFFLOAD_DEVICE_ATTR __attribute__((always_inline))    \
    unsigned int                                                               \
    __get_x() __GPU_COORD_GETTER(__fx)                                         \
    static inline __LLVM_OFFLOAD_DEVICE_ATTR __attribute__((always_inline))    \
    unsigned int                                                               \
    __get_y() __GPU_COORD_GETTER(__fy)                                         \
    static inline __LLVM_OFFLOAD_DEVICE_ATTR __attribute__((always_inline))    \
    unsigned int                                                               \
    __get_z() __GPU_COORD_GETTER(__fz)                                         \
                                                                               \
  private:                                                                     \
    __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag);                                   \
  }
#else
#define __GPU_COORD_BUILTIN(__tag, __fx, __fy, __fz)                           \
  struct __tag {                                                               \
    unsigned int x;                                                            \
    unsigned int y;                                                            \
    unsigned int z;                                                            \
                                                                               \
  private:                                                                     \
    __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag);                                   \
  }
#endif

__GPU_COORD_BUILTIN(__gpu_builtin_threadIdx_t, __gpu_thread_id_x(),
                    __gpu_thread_id_y(), __gpu_thread_id_z());
__GPU_COORD_BUILTIN(__gpu_builtin_blockIdx_t, __gpu_block_id_x(),
                    __gpu_block_id_y(), __gpu_block_id_z());
__GPU_COORD_BUILTIN(__gpu_builtin_blockDim_t, __gpu_num_threads_x(),
                    __gpu_num_threads_y(), __gpu_num_threads_z());
__GPU_COORD_BUILTIN(__gpu_builtin_gridDim_t, __gpu_num_blocks_x(),
                    __gpu_num_blocks_y(), __gpu_num_blocks_z());

#if __LLVM_OFFLOAD_HAS_GPU_INTRINSICS
#pragma pop_macro("__GPU_COORD_GETTER")
#endif
#pragma pop_macro("__GPU_COORD_BUILTIN")
#pragma pop_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")

extern const __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __gpu_builtin_threadIdx_t threadIdx;
extern const __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __gpu_builtin_blockIdx_t blockIdx;
extern const __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __gpu_builtin_blockDim_t blockDim;
extern const __LLVM_OFFLOAD_DEVICE_WEAK_ATTR __gpu_builtin_gridDim_t gridDim;

// warpSize: reads the actual warp/wavefront size from hardware
// Uses implicit conversion operator to allow direct use as int.
#if defined(__NVPTX__) || defined(__AMDGPU__) || defined(__SPIRV__)
#pragma push_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")
#define __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag)                                \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag() = delete;                                 \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag(const __tag &) = delete;                    \
  __LLVM_OFFLOAD_DEVICE_ATTR void operator=(const __tag &) const = delete;     \
  __LLVM_OFFLOAD_DEVICE_ATTR __tag *operator&() const = delete

struct __gpu_builtin_warpSize_t {
  __LLVM_OFFLOAD_DEVICE_ATTR constexpr operator int() const {
#if __LLVM_OFFLOAD_HAS_GPU_INTRINSICS
    return __gpu_num_lanes();
#endif
  }

private:
  __GPU_DISALLOW_BUILTINVAR_ACCESS(__gpu_builtin_warpSize_t);
};
#pragma pop_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")

// Provide an inline definition instead of just extern declaration
static const __LLVM_OFFLOAD_DEVICE_ATTR __gpu_builtin_warpSize_t warpSize{};
#endif

#undef __LLVM_OFFLOAD_HAS_GPU_INTRINSICS

#undef __LLVM_OFFLOAD_DEVICE_ATTR
#undef __LLVM_OFFLOAD_DEVICE_WEAK_ATTR
