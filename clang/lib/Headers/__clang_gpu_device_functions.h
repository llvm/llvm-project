//===---- __clang_gpu_device_functions.h - GPU device functions ------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_GPU_DEVICE_FUNCTIONS_H__
#define __CLANG_GPU_DEVICE_FUNCTIONS_H__

#if defined(__HIP__) || defined(__CUDA__)

#ifndef __device__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#endif

#include <gpuintrin.h>

#pragma push_macro("__GPU_DEVICE__")
#define __GPU_DEVICE__ static __inline__ __attribute__((device, always_inline))

#pragma push_macro("MAYBE_UNDEF")
#define MAYBE_UNDEF __attribute__((maybe_undef))

// warpSize and the threadIdx/blockIdx/blockDim/gridDim coordinate variables.
#include <__clang_gpu_builtin_vars.h>

//===----------------------------------------------------------------------===//
// Integer intrinsics.
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ unsigned int __popc(unsigned int __x) {
  return __builtin_popcountg(__x);
}
__GPU_DEVICE__ unsigned int __popcll(unsigned long long __x) {
  return __builtin_popcountg(__x);
}

__GPU_DEVICE__ int __clz(int __x) {
  return __builtin_clzg((unsigned int)__x, 32);
}
__GPU_DEVICE__ int __clzll(long long __x) {
  return __builtin_clzg((unsigned long long)__x, 64);
}

__GPU_DEVICE__ int __ffs(int __x) {
  return __builtin_ctzg((unsigned int)__x, -1) + 1;
}
__GPU_DEVICE__ int __ffs(unsigned int __x) {
  return __builtin_ctzg(__x, -1) + 1;
}
__GPU_DEVICE__ int __ffsll(long long __x) {
  return __builtin_ctzg((unsigned long long)__x, -1) + 1;
}
__GPU_DEVICE__ int __ffsll(unsigned long long __x) {
  return __builtin_ctzg(__x, -1) + 1;
}

__GPU_DEVICE__ unsigned int __brev(unsigned int __x) {
  return __builtin_elementwise_bitreverse(__x);
}
__GPU_DEVICE__ unsigned long long __brevll(unsigned long long __x) {
  return __builtin_elementwise_bitreverse(__x);
}

__GPU_DEVICE__ int __mul24(int __x, int __y) {
  return ((int(unsigned(__x) << 8) >> 8)) * ((int(unsigned(__y) << 8) >> 8));
}
__GPU_DEVICE__ int __umul24(unsigned int __x, unsigned int __y) {
  return int((__x & 0x00ffffffu) * (__y & 0x00ffffffu));
}

__GPU_DEVICE__ int __mulhi(int __x, int __y) {
  return int(((long long)__x * (long long)__y) >> 32);
}
__GPU_DEVICE__ unsigned int __umulhi(unsigned int __x, unsigned int __y) {
  return (unsigned int)(((unsigned long long)__x * (unsigned long long)__y) >>
                        32);
}
__GPU_DEVICE__ long long __mul64hi(long long __x, long long __y) {
  return (long long)(((__int128)__x * (__int128)__y) >> 64);
}
__GPU_DEVICE__ unsigned long long __umul64hi(unsigned long long __x,
                                             unsigned long long __y) {
  return (
      unsigned long long)(((unsigned __int128)__x * (unsigned __int128)__y) >>
                          64);
}

__GPU_DEVICE__ unsigned int __sad(int __x, int __y, unsigned int __z) {
  return __x > __y ? __x - __y + __z : __y - __x + __z;
}
__GPU_DEVICE__ unsigned int __usad(unsigned int __x, unsigned int __y,
                                   unsigned int __z) {
  return __x > __y ? __x - __y + __z : __y - __x + __z;
}

__GPU_DEVICE__ int __hadd(int __x, int __y) {
  return int(((long long)__x + (long long)__y) >> 1);
}
__GPU_DEVICE__ int __rhadd(int __x, int __y) {
  return int(((long long)__x + (long long)__y + 1) >> 1);
}
__GPU_DEVICE__ unsigned int __uhadd(unsigned int __x, unsigned int __y) {
  return (unsigned int)(((unsigned long long)__x + (unsigned long long)__y) >>
                        1);
}
__GPU_DEVICE__ unsigned int __urhadd(unsigned int __x, unsigned int __y) {
  return (
      unsigned int)(((unsigned long long)__x + (unsigned long long)__y + 1) >>
                    1);
}

__GPU_DEVICE__ unsigned int __byte_perm(unsigned int __x, unsigned int __y,
                                        unsigned int __s) {
  unsigned long long __tmp = ((unsigned long long)__y << 32) | __x;
  unsigned int __result = 0;
  for (int __i = 0; __i < 4; ++__i) {
    unsigned int __sel = (__s >> (__i * 4)) & 0x7u;
    __result |= (unsigned int)((__tmp >> (__sel * 8)) & 0xffu) << (__i * 8);
  }
  return __result;
}

//===----------------------------------------------------------------------===//
// Bitfield operations.
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ unsigned int __lastbit_u32_u64(unsigned long long __x) {
  return (unsigned int)__builtin_ctzg(__x, -1);
}

__GPU_DEVICE__ unsigned int __bitextract_u32(unsigned int __src,
                                             unsigned int __offset,
                                             unsigned int __width) {
  unsigned int __o = __offset & 31u;
  unsigned int __w = __width & 31u;
  return __w == 0 ? 0u : (__src << (32u - __o - __w)) >> (32u - __w);
}
__GPU_DEVICE__ unsigned long long __bitextract_u64(unsigned long long __src,
                                                   unsigned int __offset,
                                                   unsigned int __width) {
  unsigned long long __o = __offset & 63u;
  unsigned long long __w = __width & 63u;
  return __w == 0 ? 0ull : (__src << (64ull - __o - __w)) >> (64ull - __w);
}

__GPU_DEVICE__ unsigned int __bitinsert_u32(unsigned int __dst,
                                            unsigned int __src,
                                            unsigned int __offset,
                                            unsigned int __width) {
  unsigned int __o = __offset & 31u;
  unsigned int __mask = (1u << (__width & 31u)) - 1u;
  return (__dst & ~(__mask << __o)) | ((__src & __mask) << __o);
}
__GPU_DEVICE__ unsigned long long __bitinsert_u64(unsigned long long __dst,
                                                  unsigned long long __src,
                                                  unsigned int __offset,
                                                  unsigned int __width) {
  unsigned long long __o = __offset & 63u;
  unsigned long long __mask = (1ull << (__width & 63u)) - 1ull;
  return (__dst & ~(__mask << __o)) | ((__src & __mask) << __o);
}

//===----------------------------------------------------------------------===//
// Type punning.
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ int __float_as_int(float __x) {
  return __builtin_bit_cast(int, __x);
}
__GPU_DEVICE__ unsigned int __float_as_uint(float __x) {
  return __builtin_bit_cast(unsigned int, __x);
}
__GPU_DEVICE__ float __int_as_float(int __x) {
  return __builtin_bit_cast(float, __x);
}
__GPU_DEVICE__ float __uint_as_float(unsigned int __x) {
  return __builtin_bit_cast(float, __x);
}
__GPU_DEVICE__ long long __double_as_longlong(double __x) {
  return __builtin_bit_cast(long long, __x);
}
__GPU_DEVICE__ double __longlong_as_double(long long __x) {
  return __builtin_bit_cast(double, __x);
}
__GPU_DEVICE__ int __double2hiint(double __x) {
  return int(__builtin_bit_cast(unsigned long long, __x) >> 32);
}
__GPU_DEVICE__ int __double2loint(double __x) {
  return int(__builtin_bit_cast(unsigned long long, __x));
}
__GPU_DEVICE__ double __hiloint2double(int __hi, int __lo) {
  return __builtin_bit_cast(double,
                            ((unsigned long long)(unsigned int)__hi << 32) |
                                (unsigned long long)(unsigned int)__lo);
}

//===----------------------------------------------------------------------===//
// Numeric conversions with explicit rounding.
//===----------------------------------------------------------------------===//

// Floating-point to integer conversions map directly onto the rounding
// builtins.
#define __GPU_CVT_FP2INT(__name, __res, __arg)                                 \
  __GPU_DEVICE__ __res __name##_rd(__arg __x) {                                \
    return (__res)__builtin_elementwise_floor(__x);                            \
  }                                                                            \
  __GPU_DEVICE__ __res __name##_rn(__arg __x) {                                \
    return (__res)__builtin_elementwise_rint(__x);                             \
  }                                                                            \
  __GPU_DEVICE__ __res __name##_ru(__arg __x) {                                \
    return (__res)__builtin_elementwise_ceil(__x);                             \
  }                                                                            \
  __GPU_DEVICE__ __res __name##_rz(__arg __x) { return (__res)__x; }

__GPU_CVT_FP2INT(__double2int, int, double)
__GPU_CVT_FP2INT(__double2uint, unsigned int, double)
__GPU_CVT_FP2INT(__double2ll, long long, double)
__GPU_CVT_FP2INT(__double2ull, unsigned long long, double)
__GPU_CVT_FP2INT(__float2int, int, float)
__GPU_CVT_FP2INT(__float2uint, unsigned int, float)
__GPU_CVT_FP2INT(__float2ll, long long, float)
__GPU_CVT_FP2INT(__float2ull, unsigned long long, float)

#undef __GPU_CVT_FP2INT

// Round-to-nearest is a plain conversion, so the '_rn' variants are exact.
//
// TODO: Directed rounding (rd/ru/rz) for integer-to-float and the narrowing
// double-to-float conversions has no portable builtin yet (need pragma STDC
// FENV_ROUND pragma), so these are stubbed.
#define __GPU_CVT_TO_F(__name, __res, __arg)                                   \
  __GPU_DEVICE__ __res __name##_rd(__arg __x) { __builtin_trap(); }            \
  __GPU_DEVICE__ __res __name##_rn(__arg __x) { return (__res)__x; }           \
  __GPU_DEVICE__ __res __name##_ru(__arg __x) { __builtin_trap(); }            \
  __GPU_DEVICE__ __res __name##_rz(__arg __x) { __builtin_trap(); }

__GPU_CVT_TO_F(__int2float, float, int)
__GPU_CVT_TO_F(__uint2float, float, unsigned int)
__GPU_CVT_TO_F(__ll2float, float, long long)
__GPU_CVT_TO_F(__ull2float, float, unsigned long long)
__GPU_CVT_TO_F(__ll2double, double, long long)
__GPU_CVT_TO_F(__ull2double, double, unsigned long long)
__GPU_CVT_TO_F(__double2float, float, double)

#undef __GPU_CVT_TO_F

// Integer to double conversions are always exact, so only round-to-nearest is
// defined by CUDA and HIP.
__GPU_DEVICE__ double __int2double_rn(int __x) { return (double)__x; }
__GPU_DEVICE__ double __uint2double_rn(unsigned int __x) { return (double)__x; }

//===----------------------------------------------------------------------===//
// Wavefront vote and lane identity.
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ unsigned int __lane_id(void) { return __gpu_lane_id(); }

__GPU_DEVICE__ unsigned long long __ballot(int __pred) {
  return __gpu_ballot(__gpu_lane_mask(), __pred);
}
__GPU_DEVICE__ unsigned long long __ballot64(int __pred) {
  return __gpu_ballot(__gpu_lane_mask(), __pred);
}
__GPU_DEVICE__ unsigned long long __activemask(void) {
  return __gpu_ballot(__gpu_lane_mask(), 1);
}

__GPU_DEVICE__ int __all(int __pred) {
  return __gpu_ballot(__gpu_lane_mask(), __pred) == __gpu_lane_mask();
}
__GPU_DEVICE__ int __any(int __pred) {
  return __gpu_ballot(__gpu_lane_mask(), __pred) != 0ull;
}

template <typename __T>
__GPU_DEVICE__ int __gpu_fns_impl(__T __mask, unsigned int __base,
                                  int __offset) {
  const int __bits = int(sizeof(__T)) * 8;
  __T __m = __mask;
  int __off = __offset;
  if (__offset == 0) {
    __m &= (__T(1) << __base);
    __off = 1;
  } else if (__offset < 0) {
    __m = __builtin_elementwise_bitreverse(__mask);
    __base = (unsigned int)(__bits - 1) - __base;
    __off = -__offset;
  }
  __m &= (~__T(0)) << __base;
  if (__builtin_popcountg(__m) < __off)
    return -1;
  int __total = 0;
  for (int __i = __bits / 2; __i > 0; __i >>= 1) {
    __T __lo = __m & ((__T(1) << __i) - 1);
    int __pcnt = __builtin_popcountg(__lo);
    if (__pcnt < __off) {
      __m >>= __i;
      __off -= __pcnt;
      __total += __i;
    } else {
      __m = __lo;
    }
  }
  return __offset < 0 ? (__bits - 1) - __total : __total;
}
__GPU_DEVICE__ int __fns64(unsigned long long __mask, unsigned int __base,
                           int __offset) {
  return __gpu_fns_impl(__mask, __base, __offset);
}
__GPU_DEVICE__ int __fns32(unsigned long long __mask, unsigned int __base,
                           int __offset) {
  return __gpu_fns_impl((unsigned int)__mask, __base, __offset);
}
__GPU_DEVICE__ int __fns(unsigned int __mask, unsigned int __base,
                         int __offset) {
  return __fns32(__mask, __base, __offset);
}

//===----------------------------------------------------------------------===//
// Synchronization and fences
//===----------------------------------------------------------------------===//

// CUDA provides __syncthreads as an NVPTX compiler builtin directly.
#if !defined(__CUDA__)
__GPU_DEVICE__ void __syncthreads(void) { __gpu_sync_threads(); }
#endif

template <typename __Fn>
__GPU_DEVICE__ int __gpu_block_reduce_impl(int __val, int __init, __Fn __op) {
  static __attribute__((shared)) int __scratch[32];
  unsigned int __lanes = __gpu_num_lanes();
  unsigned int __nthreads = __gpu_num_threads(__GPU_X_DIM) *
                            __gpu_num_threads(__GPU_Y_DIM) *
                            __gpu_num_threads(__GPU_Z_DIM);
  unsigned int __nwarps = (__nthreads + __lanes - 1) / __lanes;
  unsigned int __tid =
      (__gpu_thread_id(__GPU_Z_DIM) * __gpu_num_threads(__GPU_Y_DIM) +
       __gpu_thread_id(__GPU_Y_DIM)) *
          __gpu_num_threads(__GPU_X_DIM) +
      __gpu_thread_id(__GPU_X_DIM);

  if (__gpu_is_first_in_lane(__gpu_lane_mask()))
    __scratch[__tid / __lanes] = __val;
  __gpu_sync_threads();

  int __acc = __init;
  for (unsigned int __i = 0; __i < __nwarps; ++__i)
    __acc = __op(__acc, __scratch[__i]);
  __gpu_sync_threads();
  return __acc;
}

__GPU_DEVICE__ int __syncthreads_count(int __pred) {
  unsigned long long __mask = __gpu_lane_mask();
  int __val = __builtin_popcountg(__gpu_ballot(__mask, __pred));
  return __gpu_block_reduce_impl(__val, 0,
                                 [](int __a, int __b) { return __a + __b; });
}
__GPU_DEVICE__ int __syncthreads_and(int __pred) {
  unsigned long long __mask = __gpu_lane_mask();
  int __val = __gpu_ballot(__mask, __pred) == __mask;
  return __gpu_block_reduce_impl(__val, 1,
                                 [](int __a, int __b) { return __a & __b; });
}
__GPU_DEVICE__ int __syncthreads_or(int __pred) {
  unsigned long long __mask = __gpu_lane_mask();
  int __val = __gpu_ballot(__mask, __pred) != 0ull;
  return __gpu_block_reduce_impl(__val, 0,
                                 [](int __a, int __b) { return __a | __b; });
}

__GPU_DEVICE__ void __threadfence(void) {
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}
__GPU_DEVICE__ void __threadfence_block(void) {
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_WRKGRP);
}
__GPU_DEVICE__ void __threadfence_system(void) {
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM);
}

//===----------------------------------------------------------------------===//
// Timers
//===----------------------------------------------------------------------===//

__GPU_DEVICE__ long long __clock64(void) {
  return (long long)__builtin_readcyclecounter();
}
__GPU_DEVICE__ long long __clock(void) { return __clock64(); }
__GPU_DEVICE__ long long clock64(void) { return __clock64(); }
__GPU_DEVICE__ long long clock(void) { return __clock(); }
__GPU_DEVICE__ long long wall_clock64(void) {
  return (long long)__builtin_readsteadycounter();
}

// Warp shuffle / synchronization / reduction intrinsics.
#include <__clang_gpu_intrinsics.h>

#pragma pop_macro("MAYBE_UNDEF")
#pragma pop_macro("__GPU_DEVICE__")

#endif // device compile
#endif // __CLANG_GPU_DEVICE_FUNCTIONS_H__
