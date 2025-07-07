/*===--- __clang_cuda_intrinsics.h - Device-side CUDA intrinsic wrappers ---===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __CLANG_CUDA_INTRINSICS_H__
#define __CLANG_CUDA_INTRINSICS_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif

// sm_30 intrinsics: __shfl_{up,down,xor}.

#define __SM_30_INTRINSICS_H__
#define __SM_30_INTRINSICS_HPP__

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#pragma push_macro("__MAKE_SHUFFLES")
#define __MAKE_SHUFFLES(__FnName, __IntIntrinsic, __FloatIntrinsic, __Mask,    \
                        __Type)                                                \
  inline __device__ int __FnName(int __val, __Type __offset,                   \
                                 int __width = warpSize) {                     \
    return __IntIntrinsic(__val, __offset,                                     \
                          ((warpSize - __width) << 8) | (__Mask));             \
  }                                                                            \
  inline __device__ float __FnName(float __val, __Type __offset,               \
                                   int __width = warpSize) {                   \
    return __FloatIntrinsic(__val, __offset,                                   \
                            ((warpSize - __width) << 8) | (__Mask));           \
  }                                                                            \
  inline __device__ unsigned int __FnName(unsigned int __val, __Type __offset, \
                                          int __width = warpSize) {            \
    return static_cast<unsigned int>(                                          \
        ::__FnName(static_cast<int>(__val), __offset, __width));               \
  }                                                                            \
  inline __device__ long long __FnName(long long __val, __Type __offset,       \
                                       int __width = warpSize) {               \
    struct __Bits {                                                            \
      int __a, __b;                                                            \
    };                                                                         \
    _Static_assert(sizeof(__val) == sizeof(__Bits));                           \
    _Static_assert(sizeof(__Bits) == 2 * sizeof(int));                         \
    __Bits __tmp;                                                              \
    memcpy(&__tmp, &__val, sizeof(__val));                                \
    __tmp.__a = ::__FnName(__tmp.__a, __offset, __width);                      \
    __tmp.__b = ::__FnName(__tmp.__b, __offset, __width);                      \
    long long __ret;                                                           \
    memcpy(&__ret, &__tmp, sizeof(__tmp));                                     \
    return __ret;                                                              \
  }                                                                            \
  inline __device__ long __FnName(long __val, __Type __offset,                 \
                                  int __width = warpSize) {                    \
    _Static_assert(sizeof(long) == sizeof(long long) ||                        \
                   sizeof(long) == sizeof(int));                               \
    if (sizeof(long) == sizeof(long long)) {                                   \
      return static_cast<long>(                                                \
          ::__FnName(static_cast<long long>(__val), __offset, __width));       \
    } else if (sizeof(long) == sizeof(int)) {                                  \
      return static_cast<long>(                                                \
          ::__FnName(static_cast<int>(__val), __offset, __width));             \
    }                                                                          \
  }                                                                            \
  inline __device__ unsigned long __FnName(                                    \
      unsigned long __val, __Type __offset, int __width = warpSize) {          \
    return static_cast<unsigned long>(                                         \
        ::__FnName(static_cast<long>(__val), __offset, __width));              \
  }                                                                            \
  inline __device__ unsigned long long __FnName(                               \
      unsigned long long __val, __Type __offset, int __width = warpSize) {     \
    return static_cast<unsigned long long>(                                    \
        ::__FnName(static_cast<long long>(__val), __offset, __width));         \
  }                                                                            \
  inline __device__ double __FnName(double __val, __Type __offset,             \
                                    int __width = warpSize) {                  \
    long long __tmp;                                                           \
    _Static_assert(sizeof(__tmp) == sizeof(__val));                            \
    memcpy(&__tmp, &__val, sizeof(__val));                                     \
    __tmp = ::__FnName(__tmp, __offset, __width);                              \
    double __ret;                                                              \
    memcpy(&__ret, &__tmp, sizeof(__ret));                                     \
    return __ret;                                                              \
  }

__MAKE_SHUFFLES(__shfl, __nvvm_shfl_idx_i32, __nvvm_shfl_idx_f32, 0x1f, int);
// We use 0 rather than 31 as our mask, because shfl.up applies to lanes >=
// maxLane.
__MAKE_SHUFFLES(__shfl_up, __nvvm_shfl_up_i32, __nvvm_shfl_up_f32, 0,
                unsigned int);
__MAKE_SHUFFLES(__shfl_down, __nvvm_shfl_down_i32, __nvvm_shfl_down_f32, 0x1f,
                unsigned int);
__MAKE_SHUFFLES(__shfl_xor, __nvvm_shfl_bfly_i32, __nvvm_shfl_bfly_f32, 0x1f,
                int);
#pragma pop_macro("__MAKE_SHUFFLES")

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#if CUDA_VERSION >= 9000
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300)
// __shfl_sync_* variants available in CUDA-9
#pragma push_macro("__MAKE_SYNC_SHUFFLES")
#define __MAKE_SYNC_SHUFFLES(__FnName, __IntIntrinsic, __FloatIntrinsic,       \
                             __Mask, __Type)                                   \
  inline __device__ int __FnName(unsigned int __mask, int __val,               \
                                 __Type __offset, int __width = warpSize) {    \
    return __IntIntrinsic(__mask, __val, __offset,                             \
                          ((warpSize - __width) << 8) | (__Mask));             \
  }                                                                            \
  inline __device__ float __FnName(unsigned int __mask, float __val,           \
                                   __Type __offset, int __width = warpSize) {  \
    return __FloatIntrinsic(__mask, __val, __offset,                           \
                            ((warpSize - __width) << 8) | (__Mask));           \
  }                                                                            \
  inline __device__ unsigned int __FnName(unsigned int __mask,                 \
                                          unsigned int __val, __Type __offset, \
                                          int __width = warpSize) {            \
    return static_cast<unsigned int>(                                          \
        ::__FnName(__mask, static_cast<int>(__val), __offset, __width));       \
  }                                                                            \
  inline __device__ long long __FnName(unsigned int __mask, long long __val,   \
                                       __Type __offset,                        \
                                       int __width = warpSize) {               \
    struct __Bits {                                                            \
      int __a, __b;                                                            \
    };                                                                         \
    _Static_assert(sizeof(__val) == sizeof(__Bits));                           \
    _Static_assert(sizeof(__Bits) == 2 * sizeof(int));                         \
    __Bits __tmp;                                                              \
    memcpy(&__tmp, &__val, sizeof(__val));                                     \
    __tmp.__a = ::__FnName(__mask, __tmp.__a, __offset, __width);              \
    __tmp.__b = ::__FnName(__mask, __tmp.__b, __offset, __width);              \
    long long __ret;                                                           \
    memcpy(&__ret, &__tmp, sizeof(__tmp));                                     \
    return __ret;                                                              \
  }                                                                            \
  inline __device__ unsigned long long __FnName(                               \
      unsigned int __mask, unsigned long long __val, __Type __offset,          \
      int __width = warpSize) {                                                \
    return static_cast<unsigned long long>(                                    \
        ::__FnName(__mask, static_cast<long long>(__val), __offset, __width)); \
  }                                                                            \
  inline __device__ long __FnName(unsigned int __mask, long __val,             \
                                  __Type __offset, int __width = warpSize) {   \
    _Static_assert(sizeof(long) == sizeof(long long) ||                        \
                   sizeof(long) == sizeof(int));                               \
    if (sizeof(long) == sizeof(long long)) {                                   \
      return static_cast<long>(::__FnName(                                     \
          __mask, static_cast<long long>(__val), __offset, __width));          \
    } else if (sizeof(long) == sizeof(int)) {                                  \
      return static_cast<long>(                                                \
          ::__FnName(__mask, static_cast<int>(__val), __offset, __width));     \
    }                                                                          \
  }                                                                            \
  inline __device__ unsigned long __FnName(                                    \
      unsigned int __mask, unsigned long __val, __Type __offset,               \
      int __width = warpSize) {                                                \
    return static_cast<unsigned long>(                                         \
        ::__FnName(__mask, static_cast<long>(__val), __offset, __width));      \
  }                                                                            \
  inline __device__ double __FnName(unsigned int __mask, double __val,         \
                                    __Type __offset, int __width = warpSize) { \
    long long __tmp;                                                           \
    _Static_assert(sizeof(__tmp) == sizeof(__val));                            \
    memcpy(&__tmp, &__val, sizeof(__val));                                     \
    __tmp = ::__FnName(__mask, __tmp, __offset, __width);                      \
    double __ret;                                                              \
    memcpy(&__ret, &__tmp, sizeof(__ret));                                     \
    return __ret;                                                              \
  }
__MAKE_SYNC_SHUFFLES(__shfl_sync, __nvvm_shfl_sync_idx_i32,
                     __nvvm_shfl_sync_idx_f32, 0x1f, int);
// We use 0 rather than 31 as our mask, because shfl.up applies to lanes >=
// maxLane.
__MAKE_SYNC_SHUFFLES(__shfl_up_sync, __nvvm_shfl_sync_up_i32,
                     __nvvm_shfl_sync_up_f32, 0, unsigned int);
__MAKE_SYNC_SHUFFLES(__shfl_down_sync, __nvvm_shfl_sync_down_i32,
                     __nvvm_shfl_sync_down_f32, 0x1f, unsigned int);
__MAKE_SYNC_SHUFFLES(__shfl_xor_sync, __nvvm_shfl_sync_bfly_i32,
                     __nvvm_shfl_sync_bfly_f32, 0x1f, int);
#pragma pop_macro("__MAKE_SYNC_SHUFFLES")

inline __device__ void __syncwarp(unsigned int mask = 0xffffffff) {
  return __nvvm_bar_warp_sync(mask);
}

inline __device__ void __barrier_sync(unsigned int id) {
  __nvvm_barrier_sync(id);
}

inline __device__ void __barrier_sync_count(unsigned int id,
                                            unsigned int count) {
  __nvvm_barrier_sync_cnt(id, count);
}

inline __device__ int __all_sync(unsigned int mask, int pred) {
  return __nvvm_vote_all_sync(mask, pred);
}

inline __device__ int __any_sync(unsigned int mask, int pred) {
  return __nvvm_vote_any_sync(mask, pred);
}

inline __device__ int __uni_sync(unsigned int mask, int pred) {
  return __nvvm_vote_uni_sync(mask, pred);
}

inline __device__ unsigned int __ballot_sync(unsigned int mask, int pred) {
  return __nvvm_vote_ballot_sync(mask, pred);
}

inline __device__ unsigned int __activemask() {
#if CUDA_VERSION < 9020
  return __nvvm_vote_ballot(1);
#else
  return __nvvm_activemask();
#endif
}

inline __device__ unsigned int __fns(unsigned mask, unsigned base, int offset) {
  return __nvvm_fns(mask, base, offset);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

// Define __match* builtins CUDA-9 headers expect to see.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
inline __device__ unsigned int __match32_any_sync(unsigned int mask,
                                                  unsigned int value) {
  return __nvvm_match_any_sync_i32(mask, value);
}

inline __device__ unsigned int
__match64_any_sync(unsigned int mask, unsigned long long value) {
  return __nvvm_match_any_sync_i64(mask, value);
}

inline __device__ unsigned int
__match32_all_sync(unsigned int mask, unsigned int value, int *pred) {
  return __nvvm_match_all_sync_i32p(mask, value, pred);
}

inline __device__ unsigned int
__match64_all_sync(unsigned int mask, unsigned long long value, int *pred) {
  return __nvvm_match_all_sync_i64p(mask, value, pred);
}
#include "crt/sm_70_rt.hpp"

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#endif // __CUDA_VERSION >= 9000

// sm_32 intrinsics: __ldg and __funnelshift_{l,lc,r,rc}.

// Prevent the vanilla sm_32 intrinsics header from being included.
#define __SM_32_INTRINSICS_H__
#define __SM_32_INTRINSICS_HPP__

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320

inline __device__ char __ldg(const char *ptr) { return __nvvm_ldg_c(ptr); }
inline __device__ short __ldg(const short *ptr) { return __nvvm_ldg_s(ptr); }
inline __device__ int __ldg(const int *ptr) { return __nvvm_ldg_i(ptr); }
inline __device__ long __ldg(const long *ptr) { return __nvvm_ldg_l(ptr); }
inline __device__ long long __ldg(const long long *ptr) {
  return __nvvm_ldg_ll(ptr);
}
inline __device__ unsigned char __ldg(const unsigned char *ptr) {
  return __nvvm_ldg_uc(ptr);
}
inline __device__ signed char __ldg(const signed char *ptr) {
  return __nvvm_ldg_uc((const unsigned char *)ptr);
}
inline __device__ unsigned short __ldg(const unsigned short *ptr) {
  return __nvvm_ldg_us(ptr);
}
inline __device__ unsigned int __ldg(const unsigned int *ptr) {
  return __nvvm_ldg_ui(ptr);
}
inline __device__ unsigned long __ldg(const unsigned long *ptr) {
  return __nvvm_ldg_ul(ptr);
}
inline __device__ unsigned long long __ldg(const unsigned long long *ptr) {
  return __nvvm_ldg_ull(ptr);
}
inline __device__ float __ldg(const float *ptr) { return __nvvm_ldg_f(ptr); }
inline __device__ double __ldg(const double *ptr) { return __nvvm_ldg_d(ptr); }

inline __device__ char2 __ldg(const char2 *ptr) {
  typedef char c2 __attribute__((ext_vector_type(2)));
  // We can assume that ptr is aligned at least to char2's alignment, but the
  // load will assume that ptr is aligned to char2's alignment.  This is only
  // safe if alignof(c2) <= alignof(char2).
  c2 rv = __nvvm_ldg_c2(reinterpret_cast<const c2 *>(ptr));
  char2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ char4 __ldg(const char4 *ptr) {
  typedef char c4 __attribute__((ext_vector_type(4)));
  c4 rv = __nvvm_ldg_c4(reinterpret_cast<const c4 *>(ptr));
  char4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ short2 __ldg(const short2 *ptr) {
  typedef short s2 __attribute__((ext_vector_type(2)));
  s2 rv = __nvvm_ldg_s2(reinterpret_cast<const s2 *>(ptr));
  short2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ short4 __ldg(const short4 *ptr) {
  typedef short s4 __attribute__((ext_vector_type(4)));
  s4 rv = __nvvm_ldg_s4(reinterpret_cast<const s4 *>(ptr));
  short4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ int2 __ldg(const int2 *ptr) {
  typedef int i2 __attribute__((ext_vector_type(2)));
  i2 rv = __nvvm_ldg_i2(reinterpret_cast<const i2 *>(ptr));
  int2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ int4 __ldg(const int4 *ptr) {
  typedef int i4 __attribute__((ext_vector_type(4)));
  i4 rv = __nvvm_ldg_i4(reinterpret_cast<const i4 *>(ptr));
  int4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ longlong2 __ldg(const longlong2 *ptr) {
  typedef long long ll2 __attribute__((ext_vector_type(2)));
  ll2 rv = __nvvm_ldg_ll2(reinterpret_cast<const ll2 *>(ptr));
  longlong2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}

inline __device__ uchar2 __ldg(const uchar2 *ptr) {
  typedef unsigned char uc2 __attribute__((ext_vector_type(2)));
  uc2 rv = __nvvm_ldg_uc2(reinterpret_cast<const uc2 *>(ptr));
  uchar2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ uchar4 __ldg(const uchar4 *ptr) {
  typedef unsigned char uc4 __attribute__((ext_vector_type(4)));
  uc4 rv = __nvvm_ldg_uc4(reinterpret_cast<const uc4 *>(ptr));
  uchar4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ ushort2 __ldg(const ushort2 *ptr) {
  typedef unsigned short us2 __attribute__((ext_vector_type(2)));
  us2 rv = __nvvm_ldg_us2(reinterpret_cast<const us2 *>(ptr));
  ushort2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ ushort4 __ldg(const ushort4 *ptr) {
  typedef unsigned short us4 __attribute__((ext_vector_type(4)));
  us4 rv = __nvvm_ldg_us4(reinterpret_cast<const us4 *>(ptr));
  ushort4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ uint2 __ldg(const uint2 *ptr) {
  typedef unsigned int ui2 __attribute__((ext_vector_type(2)));
  ui2 rv = __nvvm_ldg_ui2(reinterpret_cast<const ui2 *>(ptr));
  uint2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ uint4 __ldg(const uint4 *ptr) {
  typedef unsigned int ui4 __attribute__((ext_vector_type(4)));
  ui4 rv = __nvvm_ldg_ui4(reinterpret_cast<const ui4 *>(ptr));
  uint4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ ulonglong2 __ldg(const ulonglong2 *ptr) {
  typedef unsigned long long ull2 __attribute__((ext_vector_type(2)));
  ull2 rv = __nvvm_ldg_ull2(reinterpret_cast<const ull2 *>(ptr));
  ulonglong2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}

inline __device__ float2 __ldg(const float2 *ptr) {
  typedef float f2 __attribute__((ext_vector_type(2)));
  f2 rv = __nvvm_ldg_f2(reinterpret_cast<const f2 *>(ptr));
  float2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}
inline __device__ float4 __ldg(const float4 *ptr) {
  typedef float f4 __attribute__((ext_vector_type(4)));
  f4 rv = __nvvm_ldg_f4(reinterpret_cast<const f4 *>(ptr));
  float4 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  ret.z = rv[2];
  ret.w = rv[3];
  return ret;
}
inline __device__ double2 __ldg(const double2 *ptr) {
  typedef double d2 __attribute__((ext_vector_type(2)));
  d2 rv = __nvvm_ldg_d2(reinterpret_cast<const d2 *>(ptr));
  double2 ret;
  ret.x = rv[0];
  ret.y = rv[1];
  return ret;
}

// TODO: Implement these as intrinsics, so the backend can work its magic on
// these.  Alternatively, we could implement these as plain C and try to get
// llvm to recognize the relevant patterns.
inline __device__ unsigned __funnelshift_l(unsigned low32, unsigned high32,
                                           unsigned shiftWidth) {
  unsigned result;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(low32), "r"(high32), "r"(shiftWidth));
  return result;
}
inline __device__ unsigned __funnelshift_lc(unsigned low32, unsigned high32,
                                            unsigned shiftWidth) {
  unsigned result;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(low32), "r"(high32), "r"(shiftWidth));
  return result;
}
inline __device__ unsigned __funnelshift_r(unsigned low32, unsigned high32,
                                           unsigned shiftWidth) {
  unsigned result;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(low32), "r"(high32), "r"(shiftWidth));
  return result;
}
inline __device__ unsigned __funnelshift_rc(unsigned low32, unsigned high32,
                                            unsigned shiftWidth) {
  unsigned ret;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"(low32), "r"(high32), "r"(shiftWidth));
  return ret;
}

#if defined(__cplusplus) && (__cplusplus >= 201103L)

#pragma push_macro("__INTRINSIC_LOAD")
#define __INTRINSIC_LOAD(__FnName, __AsmOp, __DeclType, __TmpType, __AsmType,  \
                         __Clobber)                                            \
  inline __device__ __DeclType __FnName(const __DeclType *__ptr) {             \
    __TmpType __ret;                                                           \
    asm(__AsmOp " %0, [%1];" : __AsmType(__ret) : "l"(__ptr)__Clobber);        \
    return (__DeclType)__ret;                                                  \
  }

#pragma push_macro("__INTRINSIC_LOAD2")
#define __INTRINSIC_LOAD2(__FnName, __AsmOp, __DeclType, __TmpType, __AsmType, \
                          __Clobber)                                           \
  inline __device__ __DeclType __FnName(const __DeclType *__ptr) {             \
    __DeclType __ret;                                                          \
    __TmpType __tmp;                                                           \
    asm(__AsmOp " {%0,%1}, [%2];"                                              \
        : __AsmType(__tmp.x), __AsmType(__tmp.y)                               \
        : "l"(__ptr)__Clobber);                                                \
    using __ElementType = decltype(__ret.x);                                   \
    __ret.x = (__ElementType)(__tmp.x);                                        \
    __ret.y = (__ElementType)__tmp.y;                                          \
    return __ret;                                                              \
  }

#pragma push_macro("__INTRINSIC_LOAD4")
#define __INTRINSIC_LOAD4(__FnName, __AsmOp, __DeclType, __TmpType, __AsmType, \
                          __Clobber)                                           \
  inline __device__ __DeclType __FnName(const __DeclType *__ptr) {             \
    __DeclType __ret;                                                          \
    __TmpType __tmp;                                                           \
    asm(__AsmOp " {%0,%1,%2,%3}, [%4];"                                        \
        : __AsmType(__tmp.x), __AsmType(__tmp.y), __AsmType(__tmp.z),          \
          __AsmType(__tmp.w)                                                   \
        : "l"(__ptr)__Clobber);                                                \
    using __ElementType = decltype(__ret.x);                                   \
    __ret.x = (__ElementType)__tmp.x;                                          \
    __ret.y = (__ElementType)__tmp.y;                                          \
    __ret.z = (__ElementType)__tmp.z;                                          \
    __ret.w = (__ElementType)__tmp.w;                                          \
    return __ret;                                                              \
  }

__INTRINSIC_LOAD(__ldcg, "ld.global.cg.s8", char, unsigned int, "=r", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.s8", signed char, unsigned int, "=r", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.s16", short, unsigned short, "=h", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.s32", int, unsigned int, "=r", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.s64", long long, unsigned long long,
                 "=l", );

__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.s8", char2, int2, "=r", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.s8", char4, int4, "=r", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.s16", short2, short2, "=h", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.s16", short4, short4, "=h", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.s32", int2, int2, "=r", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.s32", int4, int4, "=r", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.s64 ", longlong2, longlong2, "=l", );

__INTRINSIC_LOAD(__ldcg, "ld.global.cg.u8", unsigned char, unsigned int,
                 "=r", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.u16", unsigned short, unsigned short,
                 "=h", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.u32", unsigned int, unsigned int,
                 "=r", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.u64", unsigned long long,
                 unsigned long long, "=l", );

__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.u8", uchar2, int2, "=r", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.u8", uchar4, int4, "=r", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.u16", ushort2, ushort2, "=h", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.u16", ushort4, ushort4, "=h", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.u32", uint2, uint2, "=r", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.u32", uint4, uint4, "=r", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.u64", ulonglong2, ulonglong2,
                  "=l", );

__INTRINSIC_LOAD(__ldcg, "ld.global.cg.f32", float, float, "=f", );
__INTRINSIC_LOAD(__ldcg, "ld.global.cg.f64", double, double, "=d", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.f32", float2, float2, "=f", );
__INTRINSIC_LOAD4(__ldcg, "ld.global.cg.v4.f32", float4, float4, "=f", );
__INTRINSIC_LOAD2(__ldcg, "ld.global.cg.v2.f64", double2, double2, "=d", );

inline __device__ long __ldcg(const long *__ptr) {
  unsigned long __ret;
  if (sizeof(long) == 8) {
    asm("ld.global.cg.s64 %0, [%1];" : "=l"(__ret) : "l"(__ptr));
  } else {
    asm("ld.global.cg.s32 %0, [%1];" : "=r"(__ret) : "l"(__ptr));
  }
  return (long)__ret;
}

__INTRINSIC_LOAD(__ldcv, "ld.global.cv.u8", unsigned char, unsigned int,
                 "=r", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.u16", unsigned short, unsigned short,
                 "=h", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.u32", unsigned int, unsigned int,
                 "=r", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.u64", unsigned long long,
                 unsigned long long, "=l", : "memory");

__INTRINSIC_LOAD(__ldcv, "ld.global.cv.s8", char, unsigned int,
                 "=r", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.s8", signed char, unsigned int,
                 "=r", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.s16", short, unsigned short,
                 "=h", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.s32", int, unsigned int,
                 "=r", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.s64", long long, unsigned long long,
                 "=l", : "memory");

__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.u8", uchar2, uint2,
                  "=r", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.u8", uchar4, uint4,
                  "=r", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.u16", ushort2, ushort2,
                  "=h", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.u16", ushort4, ushort4,
                  "=h", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.u32", uint2, uint2,
                  "=r", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.u32", uint4, uint4,
                  "=r", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.u64", ulonglong2, ulonglong2,
                  "=l", : "memory");

__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.s8", char2, int2, "=r", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.s8", char4, int4, "=r", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.s16", short2, short2,
                  "=h", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.s16", short4, short4,
                  "=h", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.s32", int2, int2, "=r", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.s32", int4, int4, "=r", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.s64", longlong2, longlong2,
                  "=l", : "memory");

__INTRINSIC_LOAD(__ldcv, "ld.global.cv.f32", float, float, "=f", : "memory");
__INTRINSIC_LOAD(__ldcv, "ld.global.cv.f64", double, double, "=d", : "memory");

__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.f32", float2, float2,
                  "=f", : "memory");
__INTRINSIC_LOAD4(__ldcv, "ld.global.cv.v4.f32", float4, float4,
                  "=f", : "memory");
__INTRINSIC_LOAD2(__ldcv, "ld.global.cv.v2.f64", double2, double2,
                  "=d", : "memory");

inline __device__ long __ldcv(const long *__ptr) {
  unsigned long __ret;
  if (sizeof(long) == 8) {
    asm("ld.global.cv.s64 %0, [%1];" : "=l"(__ret) : "l"(__ptr));
  } else {
    asm("ld.global.cv.s32 %0, [%1];" : "=r"(__ret) : "l"(__ptr));
  }
  return (long)__ret;
}

__INTRINSIC_LOAD(__ldcs, "ld.global.cs.s8", char, unsigned int, "=r", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.s8", signed char, signed int, "=r", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.s16", short, unsigned short, "=h", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.s32", int, unsigned int, "=r", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.s64", long long, unsigned long long,
                 "=l", );

__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.s8", char2, int2, "=r", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.s8", char4, int4, "=r", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.s16", short2, short2, "=h", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.s16", short4, short4, "=h", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.s32", int2, int2, "=r", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.s32", int4, int4, "=r", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.s64", longlong2, longlong2, "=l", );

__INTRINSIC_LOAD(__ldcs, "ld.global.cs.u8", unsigned char, unsigned int,
                 "=r", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.u16", unsigned short, unsigned short,
                 "=h", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.u32", unsigned int, unsigned int,
                 "=r", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.u64", unsigned long long,
                 unsigned long long, "=l", );

__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.u8", uchar2, uint2, "=r", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.u8", uchar4, uint4, "=r", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.u16", ushort2, ushort2, "=h", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.u16", ushort4, ushort4, "=h", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.u32", uint2, uint2, "=r", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.u32", uint4, uint4, "=r", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.u64", ulonglong2, ulonglong2,
                  "=l", );

__INTRINSIC_LOAD(__ldcs, "ld.global.cs.f32", float, float, "=f", );
__INTRINSIC_LOAD(__ldcs, "ld.global.cs.f64", double, double, "=d", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.f32", float2, float2, "=f", );
__INTRINSIC_LOAD4(__ldcs, "ld.global.cs.v4.f32", float4, float4, "=f", );
__INTRINSIC_LOAD2(__ldcs, "ld.global.cs.v2.f64", double2, double2, "=d", );

#pragma pop_macro("__INTRINSIC_LOAD")
#pragma pop_macro("__INTRINSIC_LOAD2")
#pragma pop_macro("__INTRINSIC_LOAD4")

inline __device__ long __ldcs(const long *__ptr) {
  unsigned long __ret;
  if (sizeof(long) == 8) {
    asm("ld.global.cs.s64 %0, [%1];" : "=l"(__ret) : "l"(__ptr));
  } else {
    asm("ld.global.cs.s32 %0, [%1];" : "=r"(__ret) : "l"(__ptr));
  }
  return (long)__ret;
}

#pragma push_macro("__INTRINSIC_STORE")
#define __INTRINSIC_STORE(__FnName, __AsmOp, __DeclType, __TmpType, __AsmType) \
  inline __device__ void __FnName(__DeclType *__ptr, __DeclType __value) {     \
    __TmpType __tmp = (__TmpType)__value;                                      \
    asm(__AsmOp " [%0], %1;" ::"l"(__ptr), __AsmType(__tmp) : "memory");       \
  }

#pragma push_macro("__INTRINSIC_STORE2")
#define __INTRINSIC_STORE2(__FnName, __AsmOp, __DeclType, __TmpType,           \
                           __AsmType)                                          \
  inline __device__ void __FnName(__DeclType *__ptr, __DeclType __value) {     \
    __TmpType __tmp;                                                           \
    using __ElementType = decltype(__tmp.x);                                   \
    __tmp.x = (__ElementType)(__value.x);                                      \
    __tmp.y = (__ElementType)(__value.y);                                      \
    asm(__AsmOp " [%0], {%1,%2};" ::"l"(__ptr), __AsmType(__tmp.x),            \
        __AsmType(__tmp.y)                                                     \
        : "memory");                                                           \
  }

#pragma push_macro("__INTRINSIC_STORE4")
#define __INTRINSIC_STORE4(__FnName, __AsmOp, __DeclType, __TmpType,           \
                           __AsmType)                                          \
  inline __device__ void __FnName(__DeclType *__ptr, __DeclType __value) {     \
    __TmpType __tmp;                                                           \
    using __ElementType = decltype(__tmp.x);                                   \
    __tmp.x = (__ElementType)(__value.x);                                      \
    __tmp.y = (__ElementType)(__value.y);                                      \
    __tmp.z = (__ElementType)(__value.z);                                      \
    __tmp.w = (__ElementType)(__value.w);                                      \
    asm(__AsmOp " [%0], {%1,%2,%3,%4};" ::"l"(__ptr), __AsmType(__tmp.x),      \
        __AsmType(__tmp.y), __AsmType(__tmp.z), __AsmType(__tmp.w)             \
        : "memory");                                                           \
  }

__INTRINSIC_STORE(__stwt, "st.global.wt.s8", char, int, "r");
__INTRINSIC_STORE(__stwt, "st.global.wt.s8", signed char, int, "r");
__INTRINSIC_STORE(__stwt, "st.global.wt.s16", short, short, "h");
__INTRINSIC_STORE(__stwt, "st.global.wt.s32", int, int, "r");
__INTRINSIC_STORE(__stwt, "st.global.wt.s64", long long, long long, "l");

__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.s8", char2, int2, "r");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.s8", char4, int4, "r");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.s16", short2, short2, "h");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.s16", short4, short4, "h");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.s32", int2, int2, "r");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.s32", int4, int4, "r");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.s64", longlong2, longlong2, "l");

__INTRINSIC_STORE(__stwt, "st.global.wt.u8", unsigned char, int, "r");
__INTRINSIC_STORE(__stwt, "st.global.wt.u16", unsigned short, unsigned short,
                  "h");
__INTRINSIC_STORE(__stwt, "st.global.wt.u32", unsigned int, unsigned int, "r");
__INTRINSIC_STORE(__stwt, "st.global.wt.u64", unsigned long long,
                  unsigned long long, "l");

__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.u8", uchar2, uchar2, "r");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.u8", uchar4, uint4, "r");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.u16", ushort2, ushort2, "h");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.u16", ushort4, ushort4, "h");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.u32", uint2, uint2, "r");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.u32", uint4, uint4, "r");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.u64", ulonglong2, ulonglong2, "l");

__INTRINSIC_STORE(__stwt, "st.global.wt.f32", float, float, "f");
__INTRINSIC_STORE(__stwt, "st.global.wt.f64", double, double, "d");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.f32", float2, float2, "f");
__INTRINSIC_STORE4(__stwt, "st.global.wt.v4.f32", float4, float4, "f");
__INTRINSIC_STORE2(__stwt, "st.global.wt.v2.f64", double2, double2, "d");

#pragma pop_macro("__INTRINSIC_STORE")
#pragma pop_macro("__INTRINSIC_STORE2")
#pragma pop_macro("__INTRINSIC_STORE4")

#endif // defined(__cplusplus) && (__cplusplus >= 201103L)
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 320

#if CUDA_VERSION >= 11000
extern "C" {
__device__ inline size_t __nv_cvta_generic_to_global_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(1))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_shared_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(3))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_constant_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(4))) *)__ptr;
}
__device__ inline size_t __nv_cvta_generic_to_local_impl(const void *__ptr) {
  return (size_t)(void __attribute__((address_space(5))) *)__ptr;
}
__device__ inline void *__nv_cvta_global_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(1))) *)__ptr;
}
__device__ inline void *__nv_cvta_shared_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(3))) *)__ptr;
}
__device__ inline void *__nv_cvta_constant_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(4))) *)__ptr;
}
__device__ inline void *__nv_cvta_local_to_generic_impl(size_t __ptr) {
  return (void *)(void __attribute__((address_space(5))) *)__ptr;
}
__device__ inline cuuint32_t __nvvm_get_smem_pointer(void *__ptr) {
  return __nv_cvta_generic_to_shared_impl(__ptr);
}
} // extern "C"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ inline unsigned __reduce_add_sync(unsigned __mask,
                                             unsigned __value) {
  return __nvvm_redux_sync_add(__value, __mask);
}
__device__ inline unsigned __reduce_min_sync(unsigned __mask,
                                             unsigned __value) {
  return __nvvm_redux_sync_umin(__value, __mask);
}
__device__ inline unsigned __reduce_max_sync(unsigned __mask,
                                             unsigned __value) {
  return __nvvm_redux_sync_umax(__value, __mask);
}
__device__ inline int __reduce_min_sync(unsigned __mask, int __value) {
  return __nvvm_redux_sync_min(__value, __mask);
}
__device__ inline int __reduce_max_sync(unsigned __mask, int __value) {
  return __nvvm_redux_sync_max(__value, __mask);
}
__device__ inline unsigned __reduce_or_sync(unsigned __mask, unsigned __value) {
  return __nvvm_redux_sync_or(__value, __mask);
}
__device__ inline unsigned __reduce_and_sync(unsigned __mask,
                                             unsigned __value) {
  return __nvvm_redux_sync_and(__value, __mask);
}
__device__ inline unsigned __reduce_xor_sync(unsigned __mask,
                                             unsigned __value) {
  return __nvvm_redux_sync_xor(__value, __mask);
}

__device__ inline void __nv_memcpy_async_shared_global_4(void *__dst,
                                                         const void *__src,
                                                         unsigned __src_size) {
  __nvvm_cp_async_ca_shared_global_4(
      (void __attribute__((address_space(3))) *)__dst,
      (const void __attribute__((address_space(1))) *)__src, __src_size);
}
__device__ inline void __nv_memcpy_async_shared_global_8(void *__dst,
                                                         const void *__src,
                                                         unsigned __src_size) {
  __nvvm_cp_async_ca_shared_global_8(
      (void __attribute__((address_space(3))) *)__dst,
      (const void __attribute__((address_space(1))) *)__src, __src_size);
}
__device__ inline void __nv_memcpy_async_shared_global_16(void *__dst,
                                                          const void *__src,
                                                          unsigned __src_size) {
  __nvvm_cp_async_ca_shared_global_16(
      (void __attribute__((address_space(3))) *)__dst,
      (const void __attribute__((address_space(1))) *)__src, __src_size);
}

__device__ inline void *
__nv_associate_access_property(const void *__ptr, unsigned long long __prop) {
  // TODO: it appears to provide compiler with some sort of a hint. We do not
  // know what exactly it is supposed to do. However, CUDA headers suggest that
  // just passing through __ptr should not affect correctness. They do so on
  // pre-sm80 GPUs where this builtin is not available.
  return (void*)__ptr;
}
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
__device__ inline unsigned __isCtaShared(const void *ptr) {
  return __isShared(ptr);
}

__device__ inline unsigned __isClusterShared(const void *__ptr) {
  return __nvvm_isspacep_shared_cluster(__ptr);
}

__device__ inline void *__cluster_map_shared_rank(const void *__ptr,
                                                  unsigned __rank) {
  return __nvvm_mapa((void *)__ptr, __rank);
}

__device__ inline unsigned __cluster_query_shared_rank(const void *__ptr) {
  return __nvvm_getctarank((void *)__ptr);
}

__device__ inline uint2
__cluster_map_shared_multicast(const void *__ptr,
                               unsigned int __cluster_cta_mask) {
  return make_uint2((unsigned)__cvta_generic_to_shared(__ptr),
                    __cluster_cta_mask);
}

__device__ inline unsigned __clusterDimIsSpecified() {
  return __nvvm_is_explicit_cluster();
}

__device__ inline dim3 __clusterDim() {
  return dim3(__nvvm_read_ptx_sreg_cluster_nctaid_x(),
              __nvvm_read_ptx_sreg_cluster_nctaid_y(),
              __nvvm_read_ptx_sreg_cluster_nctaid_z());
}

__device__ inline dim3 __clusterRelativeBlockIdx() {
  return dim3(__nvvm_read_ptx_sreg_cluster_ctaid_x(),
              __nvvm_read_ptx_sreg_cluster_ctaid_y(),
              __nvvm_read_ptx_sreg_cluster_ctaid_z());
}

__device__ inline dim3 __clusterGridDimInClusters() {
  return dim3(__nvvm_read_ptx_sreg_nclusterid_x(),
              __nvvm_read_ptx_sreg_nclusterid_y(),
              __nvvm_read_ptx_sreg_nclusterid_z());
}

__device__ inline dim3 __clusterIdx() {
  return dim3(__nvvm_read_ptx_sreg_clusterid_x(),
              __nvvm_read_ptx_sreg_clusterid_y(),
              __nvvm_read_ptx_sreg_clusterid_z());
}

__device__ inline unsigned __clusterRelativeBlockRank() {
  return __nvvm_read_ptx_sreg_cluster_ctarank();
}

__device__ inline unsigned __clusterSizeInBlocks() {
  return __nvvm_read_ptx_sreg_cluster_nctarank();
}

__device__ inline void __cluster_barrier_arrive() {
  __nvvm_barrier_cluster_arrive();
}

__device__ inline void __cluster_barrier_arrive_relaxed() {
  __nvvm_barrier_cluster_arrive_relaxed();
}

__device__ inline void __cluster_barrier_wait() {
  __nvvm_barrier_cluster_wait();
}

__device__ inline void __threadfence_cluster() { __nvvm_fence_sc_cluster(); }

__device__ inline float2 atomicAdd(float2 *__ptr, float2 __val) {
  float2 __ret;
  __asm__("atom.add.v2.f32         {%0, %1}, [%2], {%3, %4};"
          : "=f"(__ret.x), "=f"(__ret.y)
          : "l"(__ptr), "f"(__val.x), "f"(__val.y));
  return __ret;
}

__device__ inline float2 atomicAdd_block(float2 *__ptr, float2 __val) {
  float2 __ret;
  __asm__("atom.cta.add.v2.f32         {%0, %1}, [%2], {%3, %4};"
          : "=f"(__ret.x), "=f"(__ret.y)
          : "l"(__ptr), "f"(__val.x), "f"(__val.y));
  return __ret;
}

__device__ inline float2 atomicAdd_system(float2 *__ptr, float2 __val) {
  float2 __ret;
  __asm__("atom.sys.add.v2.f32         {%0, %1}, [%2], {%3, %4};"
          : "=f"(__ret.x), "=f"(__ret.y)
          : "l"(__ptr), "f"(__val.x), "f"(__val.y));
  return __ret;
}

__device__ inline float4 atomicAdd(float4 *__ptr, float4 __val) {
  float4 __ret;
  __asm__("atom.add.v4.f32         {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8};"
          : "=f"(__ret.x), "=f"(__ret.y), "=f"(__ret.z), "=f"(__ret.w)
          : "l"(__ptr), "f"(__val.x), "f"(__val.y), "f"(__val.z), "f"(__val.w));
  return __ret;
}

__device__ inline float4 atomicAdd_block(float4 *__ptr, float4 __val) {
  float4 __ret;
  __asm__(
      "atom.cta.add.v4.f32         {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8};"
      : "=f"(__ret.x), "=f"(__ret.y), "=f"(__ret.z), "=f"(__ret.w)
      : "l"(__ptr), "f"(__val.x), "f"(__val.y), "f"(__val.z), "f"(__val.w));
  return __ret;
}

__device__ inline float4 atomicAdd_system(float4 *__ptr, float4 __val) {
  float4 __ret;
  __asm__(
      "atom.sys.add.v4.f32         {%0, %1, %2, %3}, [%4], {%5, %6, %7, %8};"
      : "=f"(__ret.x), "=f"(__ret.y), "=f"(__ret.z), "=f"(__ret.w)
      : "l"(__ptr), "f"(__val.x), "f"(__val.y), "f"(__val.z), "f"(__val.w)
      :);
  return __ret;
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
#endif // CUDA_VERSION >= 11000

#endif // defined(__CLANG_CUDA_INTRINSICS_H__)
