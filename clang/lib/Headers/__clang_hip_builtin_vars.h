//===---- __clang_hip_builtin_vars.h - HIP built-in variables --------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_HIP_BUILTIN_VARS_H__
#define __CLANG_HIP_BUILTIN_VARS_H__

#if __HIP__ && (defined(__AMDGPU__))

#include <gpuintrin.h>

// The warpSize is a runtime value rather than a compile-time constant.
inline __attribute__((device)) const struct {
  __attribute__((device, always_inline, const)) operator int() const noexcept {
    return __gpu_num_lanes();
  }
} warpSize{};

#pragma push_macro("__HIP_COORD_BUILTIN")
#define __HIP_COORD_BUILTIN(__tag, __fx, __fy, __fz)                           \
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
  }

__HIP_COORD_BUILTIN(__hip_builtin_threadIdx_t, __gpu_thread_id_x(),
                    __gpu_thread_id_y(), __gpu_thread_id_z());
__HIP_COORD_BUILTIN(__hip_builtin_blockIdx_t, __gpu_block_id_x(),
                    __gpu_block_id_y(), __gpu_block_id_z());
__HIP_COORD_BUILTIN(__hip_builtin_blockDim_t, __gpu_num_threads_x(),
                    __gpu_num_threads_y(), __gpu_num_threads_z());
__HIP_COORD_BUILTIN(__hip_builtin_gridDim_t, __gpu_num_blocks_x(),
                    __gpu_num_blocks_y(), __gpu_num_blocks_z());

#pragma pop_macro("__HIP_COORD_BUILTIN")

extern const __attribute__((device, weak)) __hip_builtin_threadIdx_t threadIdx;
extern const __attribute__((device, weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __attribute__((device, weak)) __hip_builtin_blockDim_t blockDim;
extern const __attribute__((device, weak)) __hip_builtin_gridDim_t gridDim;

#endif // __HIP__ && (defined(__AMDGPU__) || defined(__SPIRV__))
#endif // __CLANG_HIP_BUILTIN_VARS_H__
