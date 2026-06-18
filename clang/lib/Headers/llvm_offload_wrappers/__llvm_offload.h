/*===------ LLVM/Offload helpers for kernel languages (CUDA/HIP) -*- c++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#include <stddef.h>

#pragma push_macro("_OPENMP")
#define _OPENMP
#include <gpuintrin.h>
#pragma pop_macro("_OPENMP")

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

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

// Make sure nobody can create instances of the coordinate types, take their
// address, copy, or assign them.
#pragma push_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")
#define __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag)                                \
  __attribute__((device)) __tag() = delete;                                    \
  __attribute__((device)) __tag(const __tag &) = delete;                       \
  __attribute__((device)) void operator=(const __tag &) const = delete;        \
  __attribute__((device)) __tag *operator&() const = delete

#pragma push_macro("__GPU_COORD_BUILTIN")
#define __GPU_COORD_BUILTIN(__tag, __fx, __fy, __fz)                           \
  struct __tag {                                                               \
    __declspec(property(get = __get_x)) unsigned int x;                        \
    __declspec(property(get = __get_y)) unsigned int y;                        \
    __declspec(property(get = __get_z)) unsigned int z;                        \
    static inline __attribute__((device, always_inline)) unsigned int          \
    __get_x() {                                                                \
      return __fx;                                                             \
    }                                                                          \
    static inline __attribute__((device, always_inline)) unsigned int          \
    __get_y() {                                                                \
      return __fy;                                                             \
    }                                                                          \
    static inline __attribute__((device, always_inline)) unsigned int          \
    __get_z() {                                                                \
      return __fz;                                                             \
    }                                                                          \
                                                                               \
  private:                                                                     \
    __GPU_DISALLOW_BUILTINVAR_ACCESS(__tag);                                   \
  }

__GPU_COORD_BUILTIN(__gpu_builtin_threadIdx_t, __gpu_thread_id_x(),
                    __gpu_thread_id_y(), __gpu_thread_id_z());
__GPU_COORD_BUILTIN(__gpu_builtin_blockIdx_t, __gpu_block_id_x(),
                    __gpu_block_id_y(), __gpu_block_id_z());
__GPU_COORD_BUILTIN(__gpu_builtin_blockDim_t, __gpu_num_threads_x(),
                    __gpu_num_threads_y(), __gpu_num_threads_z());
__GPU_COORD_BUILTIN(__gpu_builtin_gridDim_t, __gpu_num_blocks_x(),
                    __gpu_num_blocks_y(), __gpu_num_blocks_z());

#pragma pop_macro("__GPU_COORD_BUILTIN")
#pragma pop_macro("__GPU_DISALLOW_BUILTINVAR_ACCESS")

extern const __attribute__((device, weak)) __gpu_builtin_threadIdx_t threadIdx;
extern const __attribute__((device, weak)) __gpu_builtin_blockIdx_t blockIdx;
extern const __attribute__((device, weak)) __gpu_builtin_blockDim_t blockDim;
extern const __attribute__((device, weak)) __gpu_builtin_gridDim_t gridDim;
