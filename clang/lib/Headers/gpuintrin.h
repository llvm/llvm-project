//===-- gpuintrin.h - Generic GPU intrinsic functions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides wrappers around the clang builtins for accessing GPU hardware
// features. The interface is intended to be portable between architectures, but
// some targets may provide different implementations. This header can be
// included for all the common GPU programming languages, namely OpenMP, HIP,
// CUDA, and OpenCL.
//
//===----------------------------------------------------------------------===//

#ifndef __GPUINTRIN_H
#define __GPUINTRIN_H

#if !defined(_DEFAULT_FN_ATTRS)
#if defined(__HIP__) || defined(__CUDA__)
#define _DEFAULT_FN_ATTRS __attribute__((device))
#else
#define _DEFAULT_FN_ATTRS
#endif
#endif

#if defined(__NVPTX__)
#include <nvptxintrin.h>
#elif defined(__AMDGPU__)
#include <amdgpuintrin.h>
#else
#error "This header is only meant to be used on GPU architectures."
#endif

#if !defined(__cplusplus)
_Pragma("push_macro(\"bool\")");
#define bool _Bool
#endif

_Pragma("omp begin declare target device_type(nohost)");

// Returns the number of blocks in the requested dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_blocks(int __dim) {
  switch (__dim) {
  case 0:
    return __gpu_num_blocks_x();
  case 1:
    return __gpu_num_blocks_y();
  case 2:
    return __gpu_num_blocks_z();
  default:
    __builtin_unreachable();
  }
}

// Returns the number of block id in the requested dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_block_id(int __dim) {
  switch (__dim) {
  case 0:
    return __gpu_block_id_x();
  case 1:
    return __gpu_block_id_y();
  case 2:
    return __gpu_block_id_z();
  default:
    __builtin_unreachable();
  }
}

// Returns the number of threads in the requested dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_num_threads(int __dim) {
  switch (__dim) {
  case 0:
    return __gpu_num_threads_x();
  case 1:
    return __gpu_num_threads_y();
  case 2:
    return __gpu_num_threads_z();
  default:
    __builtin_unreachable();
  }
}

// Returns the thread id in the requested dimension.
_DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_thread_id(int __dim) {
  switch (__dim) {
  case 0:
    return __gpu_thread_id_x();
  case 1:
    return __gpu_thread_id_y();
  case 2:
    return __gpu_thread_id_z();
  default:
    __builtin_unreachable();
  }
}

// Get the first active thread inside the lane.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_first_lane_id(uint64_t __lane_mask) {
  return __builtin_ffsll(__lane_mask) - 1;
}

// Conditional that is only true for a single thread in a lane.
_DEFAULT_FN_ATTRS static __inline__ bool
__gpu_is_first_in_lane(uint64_t __lane_mask) {
  return __gpu_lane_id() == __gpu_first_lane_id(__lane_mask);
}

// Gets the sum of all lanes inside the warp or wavefront.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_lane_reduce_u32(uint64_t __lane_mask, uint32_t x) {
  for (uint32_t step = __gpu_num_lanes() / 2; step > 0; step /= 2) {
    uint32_t index = step + __gpu_lane_id();
    x += __gpu_shuffle_idx_u32(__lane_mask, index, x);
  }
  return __gpu_read_first_lane_u32(__lane_mask, x);
}

// Gets the accumulator scan of the threads in the warp or wavefront.
_DEFAULT_FN_ATTRS static __inline__ uint32_t
__gpu_lane_scan_u32(uint64_t __lane_mask, uint32_t x) {
  for (uint32_t step = 1; step < __gpu_num_lanes(); step *= 2) {
    uint32_t index = __gpu_lane_id() - step;
    uint32_t bitmask = __gpu_lane_id() >= step;
    x += -bitmask & __gpu_shuffle_idx_u32(__lane_mask, index, x);
  }
  return x;
}

_Pragma("omp end declare target");

#if !defined(__cplusplus)
_Pragma("pop_macro(\"bool\")");
#endif

#undef _DEFAULT_FN_ATTRS

#endif // __GPUINTRIN_H
