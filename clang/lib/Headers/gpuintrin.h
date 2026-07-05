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

#include <stdint.h>

#if !defined(__cplusplus)
_Pragma("push_macro(\"bool\")");
#define bool _Bool
#endif

#if defined(__NVPTX__)
#include <nvptxintrin.h>
#elif defined(__AMDGPU__)
#include <amdgpuintrin.h>
#elif defined(__SPIRV__)
#include <spirvintrin.h>
#elif !defined(_OPENMP)
#error "This header is only meant to be used on GPU architectures."
#endif

_Pragma("omp begin declare target device_type(nohost)");
_Pragma("omp begin declare variant match(device = {kind(gpu)})");

// Attribute to declare a function as a kernel.
#define __gpu_kernel __attribute__((device_kernel, visibility("protected")))

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
    return 1;
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
    return 0;
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
    return 1;
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
    return 0;
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

// Copies the value from the first active thread to the rest.
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_read_first_lane_u64(uint64_t __lane_mask, uint64_t __x) {
  uint32_t __hi = (uint32_t)(__x >> 32);
  uint32_t __lo = (uint32_t)(__x & 0xFFFFFFFF);
  return ((uint64_t)__gpu_read_first_lane_u32(__lane_mask, __hi) << 32) |
         ((uint64_t)__gpu_read_first_lane_u32(__lane_mask, __lo) & 0xFFFFFFFF);
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
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_shuffle_idx_u64(uint64_t __lane_mask, uint32_t __idx, uint64_t __x,
                      uint32_t __width) {
  uint32_t __hi = (uint32_t)(__x >> 32);
  uint32_t __lo = (uint32_t)(__x & 0xFFFFFFFF);
  uint32_t __mask = (uint32_t)__lane_mask;
  return ((uint64_t)__gpu_shuffle_idx_u32(__mask, __idx, __hi, __width) << 32) |
         ((uint64_t)__gpu_shuffle_idx_u32(__mask, __idx, __lo, __width));
}

// Shuffles the the lanes according to the given index.
_DEFAULT_FN_ATTRS static __inline__ float
__gpu_shuffle_idx_f32(uint64_t __lane_mask, uint32_t __idx, float __x,
                      uint32_t __width) {
  return __builtin_bit_cast(
      float, __gpu_shuffle_idx_u32(__lane_mask, __idx,
                                   __builtin_bit_cast(uint32_t, __x), __width));
}

// Shuffles the the lanes according to the given index.
_DEFAULT_FN_ATTRS static __inline__ double
__gpu_shuffle_idx_f64(uint64_t __lane_mask, uint32_t __idx, double __x,
                      uint32_t __width) {
  return __builtin_bit_cast(
      double,
      __gpu_shuffle_idx_u64(__lane_mask, __idx,
                            __builtin_bit_cast(uint64_t, __x), __width));
}

// Implements scan and reduction operations across a GPU warp or wavefront.
//
// Both scans work by iterating log2(N) steps. The bitmask tracks the currently
// unprocessed lanes, above or below the current lane in the case of a suffix or
// prefix scan. Each iteration we shuffle in the unprocessed neighbors and then
// clear the bits that this operation handled.
#define __DO_LANE_OPS(__type, __op, __identity, __prefix, __suffix)            \
  _DEFAULT_FN_ATTRS static __inline__ __type                                   \
  __gpu_suffix_scan_##__prefix##_##__suffix(uint64_t __lane_mask,              \
                                            __type __x) {                      \
    uint64_t __above = __lane_mask & -(UINT64_C(2) << __gpu_lane_id());        \
    for (uint32_t __step = 1; __step < __gpu_num_lanes(); __step *= 2) {       \
      uint32_t __src = __builtin_ctzg(__above, (int)sizeof(__above) * 8);      \
      __type __result = __gpu_shuffle_idx_##__suffix(__lane_mask, __src, __x,  \
                                                     __gpu_num_lanes());       \
      __x = __op(__x, __above ? __result : (__type)__identity);                \
      for (uint32_t __i = 0; __i < __step; ++__i)                              \
        __above &= __above - 1;                                                \
    }                                                                          \
    return __x;                                                                \
  }                                                                            \
                                                                               \
  _DEFAULT_FN_ATTRS static __inline__ __type                                   \
  __gpu_prefix_scan_##__prefix##_##__suffix(uint64_t __lane_mask,              \
                                            __type __x) {                      \
    uint64_t __below = __lane_mask & ((UINT64_C(1) << __gpu_lane_id()) - 1);   \
    for (uint32_t __step = 1; __step < __gpu_num_lanes(); __step *= 2) {       \
      uint32_t __src = 63 - __builtin_clzg(__below, (int)sizeof(__below) * 8); \
      __type __result = __gpu_shuffle_idx_##__suffix(__lane_mask, __src, __x,  \
                                                     __gpu_num_lanes());       \
      __x = __op(__x, __below ? __result : (__type)__identity);                \
      for (uint32_t __i = 0; __i < __step; ++__i)                              \
        __below ^=                                                             \
            (UINT64_C(1) << (63 - __builtin_clzg(__below, 0))) & __below;      \
    }                                                                          \
    return __x;                                                                \
  }                                                                            \
                                                                               \
  _DEFAULT_FN_ATTRS static __inline__ __type                                   \
  __gpu_lane_##__prefix##_##__suffix(uint64_t __lane_mask, __type __x) {       \
    return __gpu_read_first_lane_##__suffix(                                   \
        __lane_mask,                                                           \
        __gpu_suffix_scan_##__prefix##_##__suffix(__lane_mask, __x));          \
  }

#define __GPU_OP(__x, __y) ((__x) + (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, 0, add, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, 0, add, u64);
__DO_LANE_OPS(float, __GPU_OP, 0, add, f32);
__DO_LANE_OPS(double, __GPU_OP, 0, add, f64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) ((__x) & (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, UINT32_MAX, and, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, UINT64_MAX, and, u64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) ((__x) | (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, 0, or, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, 0, or, u64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) ((__x) ^ (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, 0, xor, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, 0, xor, u64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) ((__x) < (__y) ? (__x) : (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, UINT32_MAX, min, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, UINT64_MAX, min, u64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) ((__x) > (__y) ? (__x) : (__y))
__DO_LANE_OPS(uint32_t, __GPU_OP, 0, max, u32);
__DO_LANE_OPS(uint64_t, __GPU_OP, 0, max, u64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) __builtin_elementwise_minnum((__x), (__y))
__DO_LANE_OPS(float, __GPU_OP, __builtin_inff(), minnum, f32);
__DO_LANE_OPS(double, __GPU_OP, __builtin_inf(), minnum, f64);
#undef __GPU_OP

#define __GPU_OP(__x, __y) __builtin_elementwise_maxnum((__x), (__y))
__DO_LANE_OPS(float, __GPU_OP, -__builtin_inff(), maxnum, f32);
__DO_LANE_OPS(double, __GPU_OP, -__builtin_inf(), maxnum, f64);
#undef __GPU_OP

#undef __DO_LANE_OPS

// Returns a bitmask marking all lanes that have the same value of __x.
#ifndef __gpu_match_any_u32_impl
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u32(uint64_t __lane_mask, uint32_t __x) {
  uint64_t __match_mask = 0;

  bool __done = 0;
  for (uint64_t __active_mask = __lane_mask; __active_mask;
       __active_mask = __gpu_ballot(__lane_mask, !__done)) {
    if (!__done) {
      uint32_t __first = __gpu_shuffle_idx_u32(
          __active_mask, __builtin_ctzg(__active_mask), __x, __gpu_num_lanes());
      uint64_t __ballot = __gpu_ballot(__active_mask, __first == __x);
      if (__first == __x) {
        __match_mask = __ballot;
        __done = 1;
      }
    }
  }
  return __match_mask;
}
#endif
#undef __gpu_match_any_u32_impl

// Returns a bitmask marking all lanes that have the same value of __x.
#ifndef __gpu_match_any_u64_impl
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_any_u64(uint64_t __lane_mask, uint64_t __x) {
  uint64_t __match_mask = 0;

  bool __done = 0;
  for (uint64_t __active_mask = __lane_mask; __active_mask;
       __active_mask = __gpu_ballot(__lane_mask, !__done)) {
    if (!__done) {
      uint64_t __first = __gpu_shuffle_idx_u64(
          __active_mask, __builtin_ctzg(__active_mask), __x, __gpu_num_lanes());
      uint64_t __ballot = __gpu_ballot(__active_mask, __first == __x);
      if (__first == __x) {
        __match_mask = __ballot;
        __done = 1;
      }
    }
  }
  return __match_mask;
}
#endif
#undef __gpu_match_any_u64_impl

// Returns the current lane mask if every lane contains __x.
#ifndef __gpu_match_all_u32_impl
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u32(uint64_t __lane_mask, uint32_t __x) {
  uint32_t __first = __gpu_shuffle_idx_u32(
      __lane_mask, __builtin_ctzg(__lane_mask), __x, __gpu_num_lanes());
  uint64_t __ballot = __gpu_ballot(__lane_mask, __x == __first);
  return __ballot == __lane_mask ? __lane_mask : UINT64_C(0);
}
#endif
#undef __gpu_match_all_u32_impl

// Returns the current lane mask if every lane contains __x.
#ifndef __gpu_match_all_u64_impl
_DEFAULT_FN_ATTRS static __inline__ uint64_t
__gpu_match_all_u64(uint64_t __lane_mask, uint64_t __x) {
  uint64_t __first = __gpu_shuffle_idx_u64(
      __lane_mask, __builtin_ctzg(__lane_mask), __x, __gpu_num_lanes());
  uint64_t __ballot = __gpu_ballot(__lane_mask, __x == __first);
  return __ballot == __lane_mask ? __lane_mask : UINT64_C(0);
}
#endif
#undef __gpu_match_all_u64_impl

_Pragma("omp end declare variant");
_Pragma("omp end declare target");

#if !defined(__cplusplus)
_Pragma("pop_macro(\"bool\")");
#endif

#undef _DEFAULT_FN_ATTRS

#endif // __GPUINTRIN_H
