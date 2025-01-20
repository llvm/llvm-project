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
#elif !defined(_OPENMP)
#error "This header is only meant to be used on GPU architectures."
#endif

#include <stdint.h>

#if !defined(__cplusplus)
_Pragma("push_macro(\"bool\")");
#define bool _Bool
#endif

_Pragma("omp begin declare target device_type(nohost)");
_Pragma("omp begin declare variant match(device = {kind(gpu)})");

#define __GPU_X_DIM 0
#define __GPU_Y_DIM 1
#define __GPU_Z_DIM 2

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

// Gets the first floating point value from the active lanes.
_DEFAULT_FN_ATTRS static __inline__ float
__gpu_read_first_lane_f32(uint64_t __lane_mask, float __x) {
  return __builtin_bit_cast(
      float, __gpu_read_first_lane_u32(__lane_mask,
                                       __builtin_bit_cast(uint32_t, __x)));
}

// Gets the first floating point value from the active lanes.
_DEFAULT_FN_ATTRS static __inline__ double
__gpu_read_first_lane_f64(uint64_t __lane_mask, double __x) {
  return __builtin_bit_cast(
      double, __gpu_read_first_lane_u64(__lane_mask,
                                        __builtin_bit_cast(uint64_t, __x)));
}

// Shuffles the the lanes according to the given index.
_DEFAULT_FN_ATTRS static __inline__ float
__gpu_shuffle_idx_f32(uint64_t __lane_mask, uint32_t __idx, float __x) {
  return __builtin_bit_cast(
      float, __gpu_shuffle_idx_u32(__lane_mask, __idx,
                                   __builtin_bit_cast(uint32_t, __x)));
}

// Shuffles the the lanes according to the given index.
_DEFAULT_FN_ATTRS static __inline__ double
__gpu_shuffle_idx_f64(uint64_t __lane_mask, uint32_t __idx, double __x) {
  return __builtin_bit_cast(
      double, __gpu_shuffle_idx_u64(__lane_mask, __idx,
                                    __builtin_bit_cast(uint64_t, __x)));
}

// Gets the sum of all lanes inside the warp or wavefront.
#define __DO_LANE_SUM(__type, __suffix)                                        \
  _DEFAULT_FN_ATTRS static __inline__ __type __gpu_lane_sum_##__suffix(        \
      uint64_t __lane_mask, __type __x) {                                      \
    for (uint32_t __step = __gpu_num_lanes() / 2; __step > 0; __step /= 2) {   \
      uint32_t __index = __step + __gpu_lane_id();                             \
      __x += __gpu_shuffle_idx_##__suffix(__lane_mask, __index, __x);          \
    }                                                                          \
    return __gpu_read_first_lane_##__suffix(__lane_mask, __x);                 \
  }
__DO_LANE_SUM(uint32_t, u32); // uint32_t __gpu_lane_sum_u32(m, x)
__DO_LANE_SUM(uint64_t, u64); // uint64_t __gpu_lane_sum_u64(m, x)
__DO_LANE_SUM(float, f32);    // float __gpu_lane_sum_f32(m, x)
__DO_LANE_SUM(double, f64);   // double __gpu_lane_sum_f64(m, x)
#undef __DO_LANE_SUM

// Gets the accumulator scan of the threads in the warp or wavefront.
#define __DO_LANE_SCAN(__type, __bitmask_type, __suffix)                       \
  _DEFAULT_FN_ATTRS static __inline__ uint32_t __gpu_lane_scan_##__suffix(     \
      uint64_t __lane_mask, uint32_t __x) {                                    \
    for (uint32_t __step = 1; __step < __gpu_num_lanes(); __step *= 2) {       \
      uint32_t __index = __gpu_lane_id() - __step;                             \
      __bitmask_type bitmask = __gpu_lane_id() >= __step;                      \
      __x += __builtin_bit_cast(                                               \
          __type,                                                              \
          -bitmask & __builtin_bit_cast(__bitmask_type,                        \
                                        __gpu_shuffle_idx_##__suffix(          \
                                            __lane_mask, __index, __x)));      \
    }                                                                          \
    return __x;                                                                \
  }
__DO_LANE_SCAN(uint32_t, uint32_t, u32); // uint32_t __gpu_lane_scan_u32(m, x)
__DO_LANE_SCAN(uint64_t, uint64_t, u64); // uint64_t __gpu_lane_scan_u64(m, x)
__DO_LANE_SCAN(float, uint32_t, f32);    // float __gpu_lane_scan_f32(m, x)
__DO_LANE_SCAN(double, uint64_t, f64);   // double __gpu_lane_scan_f64(m, x)
#undef __DO_LANE_SCAN

_Pragma("omp end declare variant");
_Pragma("omp end declare target");

#if !defined(__cplusplus)
_Pragma("pop_macro(\"bool\")");
#endif

#undef _DEFAULT_FN_ATTRS

#endif // __GPUINTRIN_H
