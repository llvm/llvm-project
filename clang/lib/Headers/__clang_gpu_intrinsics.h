//===--- __clang_gpu_intrinsics.h - Device-side GPU intrinsic wrappers ------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_GPU_INTRINSICS_H__
#define __CLANG_GPU_INTRINSICS_H__

#if defined(__HIP__) || defined(__CUDA__)

#ifndef __GPU_DEVICE__
#error                                                                         \
    "__clang_gpu_intrinsics.h must be included via __clang_gpu_device_functions.h"
#endif

//===----------------------------------------------------------------------===//
// Wavefront shuffles
//===----------------------------------------------------------------------===//

template <typename __T>
__GPU_DEVICE__ __T __gpu_shuffle_idx_impl(__T __v, unsigned int __idx,
                                          int __w) {
  if constexpr (sizeof(__T) == sizeof(unsigned long long)) {
    return __builtin_bit_cast(
        __T, __gpu_shuffle_idx_u64(__gpu_lane_mask(), __idx,
                                   __builtin_bit_cast(unsigned long long, __v),
                                   (unsigned int)__w));
  } else {
    return __builtin_bit_cast(
        __T, __gpu_shuffle_idx_u32(__gpu_lane_mask(), __idx,
                                   __builtin_bit_cast(unsigned int, __v),
                                   (unsigned int)__w));
  }
}

template <typename __T>
__GPU_DEVICE__ __T __shfl(MAYBE_UNDEF __T __var, int __src_lane,
                          int __width = warpSize) {
  return __gpu_shuffle_idx_impl(
      __var, (unsigned int)(__src_lane & (__width - 1)), __width);
}
template <typename __T>
__GPU_DEVICE__ __T __shfl_up(MAYBE_UNDEF __T __var, unsigned int __delta,
                             int __width = warpSize) {
  int __rel = int(__gpu_lane_id() & (unsigned int)(__width - 1));
  int __tgt = __rel - int(__delta);
  return __gpu_shuffle_idx_impl(
      __var, (unsigned int)(__tgt < 0 ? __rel : __tgt), __width);
}
template <typename __T>
__GPU_DEVICE__ __T __shfl_down(MAYBE_UNDEF __T __var, unsigned int __delta,
                               int __width = warpSize) {
  int __rel = int(__gpu_lane_id() & (unsigned int)(__width - 1));
  int __tgt = __rel + int(__delta);
  return __gpu_shuffle_idx_impl(
      __var, (unsigned int)(__tgt >= __width ? __rel : __tgt), __width);
}
template <typename __T>
__GPU_DEVICE__ __T __shfl_xor(MAYBE_UNDEF __T __var, int __lane_mask,
                              int __width = warpSize) {
  int __rel = int(__gpu_lane_id() & (unsigned int)(__width - 1));
  int __tgt = __rel ^ __lane_mask;
  return __gpu_shuffle_idx_impl(
      __var, (unsigned int)(__tgt >= __width ? __rel : __tgt), __width);
}

//===----------------------------------------------------------------------===//
// Warp synchronization
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ void __syncwarp(unsigned long long __mask = -1) {
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_WVFRNT);
  __gpu_sync_lane(__mask);
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_WVFRNT);
}

//===----------------------------------------------------------------------===//
// Wave syncrhonization sync aliases.
//===----------------------------------------------------------------------===//

template <typename __MaskT>
__GPU_DEVICE__ unsigned long long __ballot_sync(__MaskT __mask, int __pred) {
  return __ballot(__pred) & (unsigned long long)__mask;
}
template <typename __MaskT>
__GPU_DEVICE__ int __all_sync(__MaskT __mask, int __pred) {
  return __ballot_sync(__mask, __pred) == (unsigned long long)__mask;
}
template <typename __MaskT>
__GPU_DEVICE__ int __any_sync(__MaskT __mask, int __pred) {
  return __ballot_sync(__mask, __pred) != 0ull;
}

template <typename __MaskT, typename __T>
__GPU_DEVICE__ __T __shfl_sync(__MaskT __mask, MAYBE_UNDEF __T __var,
                               int __src_lane, int __width = warpSize) {
  (void)__mask;
  return __shfl(__var, __src_lane, __width);
}
template <typename __MaskT, typename __T>
__GPU_DEVICE__ __T __shfl_up_sync(__MaskT __mask, MAYBE_UNDEF __T __var,
                                  unsigned int __delta,
                                  int __width = warpSize) {
  (void)__mask;
  return __shfl_up(__var, __delta, __width);
}
template <typename __MaskT, typename __T>
__GPU_DEVICE__ __T __shfl_down_sync(__MaskT __mask, MAYBE_UNDEF __T __var,
                                    unsigned int __delta,
                                    int __width = warpSize) {
  (void)__mask;
  return __shfl_down(__var, __delta, __width);
}
template <typename __MaskT, typename __T>
__GPU_DEVICE__ __T __shfl_xor_sync(__MaskT __mask, MAYBE_UNDEF __T __var,
                                   int __lane_mask, int __width = warpSize) {
  (void)__mask;
  return __shfl_xor(__var, __lane_mask, __width);
}

//===----------------------------------------------------------------------===//
// Match primitives.
//===----------------------------------------------------------------------===//

template <typename __T>
__GPU_DEVICE__ unsigned long long __match_any(__T __value) {
  if constexpr (sizeof(__T) == sizeof(unsigned long long)) {
    return __gpu_match_any_u64(__gpu_lane_mask(),
                               __builtin_bit_cast(unsigned long long, __value));
  } else {
    return __gpu_match_any_u32(__gpu_lane_mask(),
                               __builtin_bit_cast(unsigned int, __value));
  }
}
template <typename __MaskT, typename __T>
__GPU_DEVICE__ unsigned long long __match_any_sync(__MaskT __mask,
                                                   __T __value) {
  return __match_any(__value) & (unsigned long long)__mask;
}

template <typename __T>
__GPU_DEVICE__ unsigned long long __match_all(__T __value, int *__pred) {
  unsigned long long __r;
  if constexpr (sizeof(__T) == sizeof(unsigned long long)) {
    __r = __gpu_match_all_u64(__gpu_lane_mask(),
                              __builtin_bit_cast(unsigned long long, __value));
  } else {
    __r = __gpu_match_all_u32(__gpu_lane_mask(),
                              __builtin_bit_cast(unsigned int, __value));
  }
  *__pred = __r != 0;
  return __r;
}
template <typename __MaskT, typename __T>
__GPU_DEVICE__ unsigned long long __match_all_sync(__MaskT __mask, __T __value,
                                                   int *__pred) {
  (void)__mask;
  return __match_all(__value, __pred);
}

//===----------------------------------------------------------------------===//
// Wave reductions.
//===----------------------------------------------------------------------===//

template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_add_sync(__MaskT __mask,
                                              unsigned int __val) {
  return __gpu_lane_add_u32((unsigned long long)__mask, __val);
}
template <typename __MaskT>
__GPU_DEVICE__ int __reduce_add_sync(__MaskT __mask, int __val) {
  return int(
      __gpu_lane_add_u32((unsigned long long)__mask, (unsigned int)__val));
}
template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_min_sync(__MaskT __mask,
                                              unsigned int __val) {
  return __gpu_lane_min_u32((unsigned long long)__mask, __val);
}
template <typename __MaskT>
__GPU_DEVICE__ int __reduce_min_sync(__MaskT __mask, int __val) {
  unsigned int __r = __gpu_lane_min_u32((unsigned long long)__mask,
                                        (unsigned int)__val ^ 0x80000000u);
  return int(__r ^ 0x80000000u);
}
template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_max_sync(__MaskT __mask,
                                              unsigned int __val) {
  return __gpu_lane_max_u32((unsigned long long)__mask, __val);
}
template <typename __MaskT>
__GPU_DEVICE__ int __reduce_max_sync(__MaskT __mask, int __val) {
  unsigned int __r = __gpu_lane_max_u32((unsigned long long)__mask,
                                        (unsigned int)__val ^ 0x80000000u);
  return int(__r ^ 0x80000000u);
}
template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_and_sync(__MaskT __mask,
                                              unsigned int __val) {
  return __gpu_lane_and_u32((unsigned long long)__mask, __val);
}
template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_or_sync(__MaskT __mask,
                                             unsigned int __val) {
  return __gpu_lane_or_u32((unsigned long long)__mask, __val);
}
template <typename __MaskT>
__GPU_DEVICE__ unsigned int __reduce_xor_sync(__MaskT __mask,
                                              unsigned int __val) {
  return __gpu_lane_xor_u32((unsigned long long)__mask, __val);
}

//===----------------------------------------------------------------------===//
// Funnel shifts.
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ unsigned int
__funnelshift_l(unsigned int __lo, unsigned int __hi, unsigned int __shift) {
  unsigned int __s = __shift & 31u;
  return (unsigned int)((((unsigned long long)__hi << 32 | __lo) << __s) >> 32);
}
__GPU_DEVICE__ unsigned int
__funnelshift_lc(unsigned int __lo, unsigned int __hi, unsigned int __shift) {
  unsigned int __s = __shift >= 32u ? 32u : __shift;
  return (unsigned int)((((unsigned long long)__hi << 32 | __lo) << __s) >> 32);
}
__GPU_DEVICE__ unsigned int
__funnelshift_r(unsigned int __lo, unsigned int __hi, unsigned int __shift) {
  unsigned int __s = __shift & 31u;
  return (unsigned int)(((unsigned long long)__hi << 32 | __lo) >> __s);
}
__GPU_DEVICE__ unsigned int
__funnelshift_rc(unsigned int __lo, unsigned int __hi, unsigned int __shift) {
  unsigned int __s = __shift >= 32u ? 32u : __shift;
  return (unsigned int)(((unsigned long long)__hi << 32 | __lo) >> __s);
}

#endif // device compile
#endif // __CLANG_GPU_INTRINSICS_H__
