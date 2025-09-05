//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/float/definitions.h>
#include <clc/opencl/as_type.h>
#include <clc/opencl/math/copysign.h>
#include <clc/opencl/math/fabs.h>
#include <clc/opencl/math/nextafter.h>
#include <clc/opencl/relational/isinf.h>
#include <clc/opencl/relational/isnan.h>
#include <clc/opencl/shared/min.h>
#include <clc/opencl/shared/vstore.h>

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define __CLC_ROUND_VEC1(out, in, ROUNDF) out = ROUNDF(in);
#define __CLC_ROUND_VEC2(out, in, ROUNDF)                                      \
  __CLC_ROUND_VEC1(out.lo, in.lo, ROUNDF);                                     \
  __CLC_ROUND_VEC1(out.hi, in.hi, ROUNDF);
#define __CLC_ROUND_VEC3(out, in, ROUNDF)                                      \
  __CLC_ROUND_VEC1(out.s0, in.s0, ROUNDF);                                     \
  __CLC_ROUND_VEC1(out.s1, in.s1, ROUNDF);                                     \
  __CLC_ROUND_VEC1(out.s2, in.s2, ROUNDF);
#define __CLC_ROUND_VEC4(out, in, ROUNDF)                                      \
  __CLC_ROUND_VEC2(out.lo, in.lo, ROUNDF);                                     \
  __CLC_ROUND_VEC2(out.hi, in.hi, ROUNDF);
#define __CLC_ROUND_VEC8(out, in, ROUNDF)                                      \
  __CLC_ROUND_VEC4(out.lo, in.lo, ROUNDF);                                     \
  __CLC_ROUND_VEC4(out.hi, in.hi, ROUNDF);
#define __CLC_ROUND_VEC16(out, in, ROUNDF)                                     \
  __CLC_ROUND_VEC8(out.lo, in.lo, ROUNDF);                                     \
  __CLC_ROUND_VEC8(out.hi, in.hi, ROUNDF);

#define __CLC_XFUNC_IMPL(SUFFIX, VEC_SIZE, TYPE, AS, ROUNDF)                   \
  void _CLC_OVERLOAD vstore_half_##VEC_SIZE(TYPE, size_t, AS half *);          \
  _CLC_OVERLOAD _CLC_DEF void vstore_half##SUFFIX(TYPE vec, size_t offset,     \
                                                  AS half *mem) {              \
    TYPE rounded_vec;                                                          \
    __CLC_ROUND_VEC##VEC_SIZE(rounded_vec, vec, ROUNDF);                       \
    vstore_half_##VEC_SIZE(rounded_vec, offset, mem);                          \
  }                                                                            \
  void _CLC_OVERLOAD vstorea_half_##VEC_SIZE(TYPE, size_t, AS half *);         \
  _CLC_OVERLOAD _CLC_DEF void vstorea_half##SUFFIX(TYPE vec, size_t offset,    \
                                                   AS half *mem) {             \
    TYPE rounded_vec;                                                          \
    __CLC_ROUND_VEC##VEC_SIZE(rounded_vec, vec, ROUNDF);                       \
    vstorea_half_##VEC_SIZE(rounded_vec, offset, mem);                         \
  }

_CLC_DEF _CLC_OVERLOAD float __clc_rtz(float x) {
  /* Handle nan corner case */
  if (isnan(x))
    return x;
  /* RTZ does not produce Inf for large numbers */
  if (fabs(x) > 65504.0f && !isinf(x))
    return copysign(65504.0f, x);

  const int exp = (as_uint(x) >> 23 & 0xff) - 127;
  /* Manage range rounded to +- zero explicitely */
  if (exp < -24)
    return copysign(0.0f, x);

  /* Remove lower 13 bits to make sure the number is rounded down */
  int mask = 0xffffe000;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14)
    mask <<= min(-(exp + 14), 10);

  return as_float(as_uint(x) & mask);
}

_CLC_DEF _CLC_OVERLOAD float __clc_rti(float x) {
  /* Handle nan corner case */
  if (isnan(x))
    return x;

  const float inf = copysign(INFINITY, x);
  uint ux = as_uint(x);

  /* Manage +- infinity explicitely */
  if (as_float(ux & 0x7fffffff) > 0x1.ffcp+15f) {
    return inf;
  }
  /* Manage +- zero explicitely */
  if ((ux & 0x7fffffff) == 0) {
    return copysign(0.0f, x);
  }

  const int exp = (as_uint(x) >> 23 & 0xff) - 127;
  /* Manage range rounded to smallest half denormal explicitely */
  if (exp < -24) {
    return copysign(0x1.0p-24f, x);
  }

  /* Set lower 13 bits */
  int mask = (1 << 13) - 1;
  /* Denormals cannot be flushed, and they use different bit for rounding */
  if (exp < -14) {
    mask = (1 << (13 + min(-(exp + 14), 10))) - 1;
  }

  const float next = nextafter(as_float(ux | mask), inf);
  return ((ux & mask) == 0) ? as_float(ux) : next;
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

#define __CLC_XFUNC(SUFFIX, VEC_SIZE, TYPE, AS)                                \
  __CLC_XFUNC_IMPL(SUFFIX, VEC_SIZE, TYPE, AS, __clc_rte)                      \
  __CLC_XFUNC_IMPL(SUFFIX##_rtz, VEC_SIZE, TYPE, AS, __clc_rtz)                \
  __CLC_XFUNC_IMPL(SUFFIX##_rtn, VEC_SIZE, TYPE, AS, __clc_rtn)                \
  __CLC_XFUNC_IMPL(SUFFIX##_rtp, VEC_SIZE, TYPE, AS, __clc_rtp)                \
  __CLC_XFUNC_IMPL(SUFFIX##_rte, VEC_SIZE, TYPE, AS, __clc_rte)

#define __CLC_FUNC(SUFFIX, VEC_SIZE, TYPE, AS)                                 \
  __CLC_XFUNC(SUFFIX, VEC_SIZE, TYPE, AS)

#define __CLC_BODY "vstore_half.inc"
#include <clc/math/gentype.inc>
#undef __CLC_FUNC
#undef __CLC_XFUNC
#undef __CLC_XFUNC_IMPL
