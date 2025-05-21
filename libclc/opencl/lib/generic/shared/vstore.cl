//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/clc.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define VSTORE_VECTORIZE(PRIM_TYPE, ADDR_SPACE)                                \
  typedef PRIM_TYPE##2 less_aligned_##ADDR_SPACE##PRIM_TYPE##2                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF void vstore2(PRIM_TYPE##2 vec, size_t offset,         \
                                      ADDR_SPACE PRIM_TYPE *mem) {             \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2                      \
           *)(&mem[2 * offset])) = vec;                                        \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF void vstore3(PRIM_TYPE##3 vec, size_t offset,         \
                                      ADDR_SPACE PRIM_TYPE *mem) {             \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##2                      \
           *)(&mem[3 * offset])) = (PRIM_TYPE##2)(vec.s0, vec.s1);             \
    mem[3 * offset + 2] = vec.s2;                                              \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##4 less_aligned_##ADDR_SPACE##PRIM_TYPE##4                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF void vstore4(PRIM_TYPE##4 vec, size_t offset,         \
                                      ADDR_SPACE PRIM_TYPE *mem) {             \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##4                      \
           *)(&mem[4 * offset])) = vec;                                        \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##8 less_aligned_##ADDR_SPACE##PRIM_TYPE##8                 \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF void vstore8(PRIM_TYPE##8 vec, size_t offset,         \
                                      ADDR_SPACE PRIM_TYPE *mem) {             \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##8                      \
           *)(&mem[8 * offset])) = vec;                                        \
  }                                                                            \
                                                                               \
  typedef PRIM_TYPE##16 less_aligned_##ADDR_SPACE##PRIM_TYPE##16               \
      __attribute__((aligned(sizeof(PRIM_TYPE))));                             \
  _CLC_OVERLOAD _CLC_DEF void vstore16(PRIM_TYPE##16 vec, size_t offset,       \
                                       ADDR_SPACE PRIM_TYPE *mem) {            \
    *((ADDR_SPACE less_aligned_##ADDR_SPACE##PRIM_TYPE##16                     \
           *)(&mem[16 * offset])) = vec;                                       \
  }

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define VSTORE_VECTORIZE_GENERIC VSTORE_VECTORIZE
#else
// The generic address space isn't available, so make the macro do nothing
#define VSTORE_VECTORIZE_GENERIC(X, Y)
#endif

#define VSTORE_ADDR_SPACES(__CLC_SCALAR___CLC_GENTYPE)                         \
  VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __private)                      \
  VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __local)                        \
  VSTORE_VECTORIZE(__CLC_SCALAR___CLC_GENTYPE, __global)                       \
  VSTORE_VECTORIZE_GENERIC(__CLC_SCALAR___CLC_GENTYPE, __generic)

VSTORE_ADDR_SPACES(char)
VSTORE_ADDR_SPACES(uchar)
VSTORE_ADDR_SPACES(short)
VSTORE_ADDR_SPACES(ushort)
VSTORE_ADDR_SPACES(int)
VSTORE_ADDR_SPACES(uint)
VSTORE_ADDR_SPACES(long)
VSTORE_ADDR_SPACES(ulong)
VSTORE_ADDR_SPACES(float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
VSTORE_ADDR_SPACES(double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
VSTORE_ADDR_SPACES(half)
#endif

#define VEC_STORE1(val, ROUNDF, BUILTIN) BUILTIN(ROUNDF(val), &mem[offset++]);

#define VEC_STORE2(val, ROUNDF, BUILTIN)                                       \
  VEC_STORE1(val.lo, ROUNDF, BUILTIN)                                          \
  VEC_STORE1(val.hi, ROUNDF, BUILTIN)
#define VEC_STORE3(val, ROUNDF, BUILTIN)                                       \
  VEC_STORE1(val.s0, ROUNDF, BUILTIN)                                          \
  VEC_STORE1(val.s1, ROUNDF, BUILTIN)                                          \
  VEC_STORE1(val.s2, ROUNDF, BUILTIN)
#define VEC_STORE4(val, ROUNDF, BUILTIN)                                       \
  VEC_STORE2(val.lo, ROUNDF, BUILTIN)                                          \
  VEC_STORE2(val.hi, ROUNDF, BUILTIN)
#define VEC_STORE8(val, ROUNDF, BUILTIN)                                       \
  VEC_STORE4(val.lo, ROUNDF, BUILTIN)                                          \
  VEC_STORE4(val.hi, ROUNDF, BUILTIN)
#define VEC_STORE16(val, ROUNDF, BUILTIN)                                      \
  VEC_STORE8(val.lo, ROUNDF, BUILTIN)                                          \
  VEC_STORE8(val.hi, ROUNDF, BUILTIN)

#define __FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, AS, ROUNDF, BUILTIN)            \
  _CLC_OVERLOAD _CLC_DEF void vstore_half##SUFFIX(TYPE vec, size_t offset,     \
                                                  AS half *mem) {              \
    offset *= VEC_SIZE;                                                        \
    VEC_STORE##VEC_SIZE(vec, ROUNDF, BUILTIN)                                  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF void vstorea_half##SUFFIX(TYPE vec, size_t offset,    \
                                                   AS half *mem) {             \
    offset *= OFFSET;                                                          \
    VEC_STORE##VEC_SIZE(vec, ROUNDF, BUILTIN)                                  \
  }

_CLC_DEF _CLC_OVERLOAD float __clc_noop(float x) { return x; }
_CLC_DEF _CLC_OVERLOAD float __clc_rtz(float x) {
  /* Remove lower 13 bits to make sure the number is rounded down */
  int mask = 0xffffe000;
  const int exp = (as_uint(x) >> 23 & 0xff) - 127;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14)
    mask <<= min(-(exp + 14), 10);
  /* RTZ does not produce Inf for large numbers */
  if (fabs(x) > 65504.0f && !isinf(x))
    return copysign(65504.0f, x);
  /* Handle nan corner case */
  if (isnan(x))
    return x;
  return as_float(as_uint(x) & mask);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rti(float x) {
  const float inf = copysign(INFINITY, x);
  /* Set lower 13 bits */
  int mask = (1 << 13) - 1;
  const int exp = (as_uint(x) >> 23 & 0xff) - 127;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14)
    mask = (1 << (13 + min(-(exp + 14), 10))) - 1;
  /* Handle nan corner case */
  if (isnan(x))
    return x;
  const float next = nextafter(as_float(as_uint(x) | mask), inf);
  return ((as_uint(x) & mask) == 0) ? x : next;
}
_CLC_DEF _CLC_OVERLOAD float __clc_rtn(float x) {
  return ((as_uint(x) & 0x80000000) == 0) ? __clc_rtz(x) : __clc_rti(x);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rtp(float x) {
  return ((as_uint(x) & 0x80000000) == 0) ? __clc_rti(x) : __clc_rtz(x);
}
_CLC_DEF _CLC_OVERLOAD float __clc_rte(float x) {
  /* Mantisa + implicit bit */
  const uint mantissa = (as_uint(x) & 0x7fffff) | (1u << 23);
  const int exp = (as_uint(x) >> 23 & 0xff) - 127;
  int shift = 13;
  if (exp < -14) {
    /* The default assumes lower 13 bits are rounded,
     * but it might be more for denormals.
     * Shifting beyond last == 0b, and qr == 00b is not necessary */
    shift += min(-(exp + 14), 15);
  }
  int mask = (1 << shift) - 1;
  const uint grs = mantissa & mask;
  const uint last = mantissa & (1 << shift);
  /* IEEE round up rule is: grs > 101b or grs == 100b and last == 1.
   * exp > 15 should round to inf. */
  bool roundup = (grs > (1 << (shift - 1))) ||
                 (grs == (1 << (shift - 1)) && last != 0) || (exp > 15);
  return roundup ? __clc_rti(x) : __clc_rtz(x);
}

#ifdef cl_khr_fp64
_CLC_DEF _CLC_OVERLOAD double __clc_noop(double x) { return x; }
_CLC_DEF _CLC_OVERLOAD double __clc_rtz(double x) {
  /* Remove lower 42 bits to make sure the number is rounded down */
  ulong mask = 0xfffffc0000000000UL;
  const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14)
    mask <<= min(-(exp + 14), 10);
  /* RTZ does not produce Inf for large numbers */
  if (fabs(x) > 65504.0 && !isinf(x))
    return copysign(65504.0, x);
  /* Handle nan corner case */
  if (isnan(x))
    return x;
  return as_double(as_ulong(x) & mask);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rti(double x) {
  const double inf = copysign((double)INFINITY, x);
  /* Set lower 42 bits */
  long mask = (1UL << 42UL) - 1UL;
  const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14)
    mask = (1UL << (42UL + min(-(exp + 14), 10))) - 1;
  /* Handle nan corner case */
  if (isnan(x))
    return x;
  const double next = nextafter(as_double(as_ulong(x) | mask), inf);
  return ((as_ulong(x) & mask) == 0) ? x : next;
}
_CLC_DEF _CLC_OVERLOAD double __clc_rtn(double x) {
  return ((as_ulong(x) & 0x8000000000000000UL) == 0) ? __clc_rtz(x)
                                                     : __clc_rti(x);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rtp(double x) {
  return ((as_ulong(x) & 0x8000000000000000UL) == 0) ? __clc_rti(x)
                                                     : __clc_rtz(x);
}
_CLC_DEF _CLC_OVERLOAD double __clc_rte(double x) {
  /* Mantisa + implicit bit */
  const ulong mantissa = (as_ulong(x) & 0xfffffffffffff) | (1UL << 52);
  const int exp = (as_ulong(x) >> 52 & 0x7ff) - 1023;
  int shift = 42;
  if (exp < -14) {
    /* The default assumes lower 13 bits are rounded,
     * but it might be more for denormals.
     * Shifting beyond last == 0b, and qr == 00b is not necessary */
    shift += min(-(exp + 14), 15);
  }
  ulong mask = (1UL << shift) - 1UL;
  const ulong grs = mantissa & mask;
  const ulong last = mantissa & (1UL << shift);
  /* IEEE round up rule is: grs > 101b or grs == 100b and last == 1.
   * exp > 15 should round to inf. */
  bool roundup = (grs > (1UL << (shift - 1UL))) ||
                 (grs == (1UL << (shift - 1UL)) && last != 0) || (exp > 15);
  return roundup ? __clc_rti(x) : __clc_rtz(x);
}
#endif

#define __XFUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, AS, BUILTIN)                   \
  __FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, AS, __clc_noop, BUILTIN)              \
  __FUNC(SUFFIX##_rtz, VEC_SIZE, OFFSET, TYPE, AS, __clc_rtz, BUILTIN)         \
  __FUNC(SUFFIX##_rtn, VEC_SIZE, OFFSET, TYPE, AS, __clc_rtn, BUILTIN)         \
  __FUNC(SUFFIX##_rtp, VEC_SIZE, OFFSET, TYPE, AS, __clc_rtp, BUILTIN)         \
  __FUNC(SUFFIX##_rte, VEC_SIZE, OFFSET, TYPE, AS, __clc_rte, BUILTIN)

#define FUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, AS, BUILTIN)                      \
  __XFUNC(SUFFIX, VEC_SIZE, OFFSET, TYPE, AS, BUILTIN)

#define __CLC_BODY "vstore_half.inc"
#include <clc/math/gentype.inc>
#undef FUNC
#undef __XFUNC
#undef __FUNC
#undef VEC_LOAD16
#undef VEC_LOAD8
#undef VEC_LOAD4
#undef VEC_LOAD3
#undef VEC_LOAD2
#undef VEC_LOAD1
#undef DECLARE_HELPER
#undef VSTORE_ADDR_SPACES
#undef VSTORE_VECTORIZE
#undef VSTORE_VECTORIZE_GENERIC
