//===---- __clang_gpu_builtin_vars.h - GPU built-in variables --------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_GPU_BUILTIN_VARS_H__
#define __CLANG_GPU_BUILTIN_VARS_H__

#if defined(__HIP__) || defined(__CUDA__)

#include <gpuintrin.h>

// The warpSize is a runtime value rather than a compile-time constant.
static inline __attribute__((device)) const struct {
  __attribute__((device, always_inline, const)) operator int() const noexcept {
    return __gpu_num_lanes();
  }
} warpSize{};

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
    __attribute__((device, always_inline)) unsigned int __get_x() const {      \
      return __fx;                                                             \
    }                                                                          \
    __attribute__((device, always_inline)) unsigned int __get_y() const {      \
      return __fy;                                                             \
    }                                                                          \
    __attribute__((device, always_inline)) unsigned int __get_z() const {      \
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

static inline const
    __attribute__((device)) __gpu_builtin_threadIdx_t threadIdx{};
static inline const __attribute__((device)) __gpu_builtin_blockIdx_t blockIdx{};
static inline const __attribute__((device)) __gpu_builtin_blockDim_t blockDim{};
static inline const __attribute__((device)) __gpu_builtin_gridDim_t gridDim{};

#endif // device compile
#endif // __CLANG_GPU_BUILTIN_VARS_H__
