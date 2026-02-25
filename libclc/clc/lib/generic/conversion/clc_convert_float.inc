//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_I2F
#ifdef __CLC_GEN_S

#if __CLC_GENSIZE_SRC == 8
#define __CLC_SRC_MAX SCHAR_MAX
#elif __CLC_GENSIZE_SRC == 16
#define __CLC_SRC_MAX SHRT_MAX
#elif __CLC_GENSIZE_SRC == 32
#define __CLC_SRC_MAX INT_MAX
#elif __CLC_GENSIZE_SRC == 64
#define __CLC_SRC_MAX LONG_MAX
#endif

#else // __CLC_GEN_S

#if __CLC_GENSIZE_SRC == 8
#define __CLC_SRC_MAX UCHAR_MAX
#elif __CLC_GENSIZE_SRC == 16
#define __CLC_SRC_MAX USHRT_MAX
#elif __CLC_GENSIZE_SRC == 32
#define __CLC_SRC_MAX UINT_MAX
#elif __CLC_GENSIZE_SRC == 64
#define __CLC_SRC_MAX ULONG_MAX
#endif

#endif // __CLC_GEN_S
#endif // __CLC_I2F

// Default conversion

// All conversions are in accordance with the OpenCL specification,
// which cites the C99 conversion rules.
//
// Conversions from integer to floating-point, and floating-point to
// floating-point through casting is done with the default rounding
// mode. While C99 allows dynamically changing the rounding mode
// during runtime, it is not a supported feature in OpenCL according
// to Section 7.1 - Rounding Modes in the OpenCL 1.2 specification.
//
// Therefore, we can assume for optimization purposes that the
// rounding mode is fixed to round-to-nearest-even. Platform target
// authors should ensure that the rounding-control registers remain
// in this state, and that this invariant holds.
//
// Also note, even though the OpenCL specification isn't entirely
// clear on this matter, we implement all rounding mode combinations
// even for integer-to-integer conversions. When such a conversion
// is used, the rounding mode is ignored.

#ifdef __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE_SRC x) {
  return (__CLC_GENTYPE)x;
}

#else // __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE_SRC x) {
  return __builtin_convertvector(x, __CLC_GENTYPE);
}

#endif // __CLC_SCALAR

// Rounding mode conversion

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,
                                                   _rte)(__CLC_GENTYPE_SRC x) {
  return __CLC_FUNCTION(x);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,
                                                   _rtz)(__CLC_GENTYPE_SRC x) {
  __CLC_GENTYPE r = __CLC_FUNCTION(x);
#ifdef __CLC_I2F
  __CLC_GENTYPE_SRC y =
      __CLC_XCONCAT(__CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC), _sat)(r);
  __CLC_GENTYPE_SRC_U abs_x = __clc_abs(x);
  __CLC_GENTYPE_SRC_U abs_y = __clc_abs(y);
#else
  __CLC_GENTYPE_SRC y = __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC)(r);
  __CLC_GENTYPE_SRC abs_x = __clc_fabs(x);
  __CLC_GENTYPE_SRC abs_y = __clc_fabs(y);
#endif
  __CLC_S_GENTYPE c = __CLC_CONVERT_S_GENTYPE(abs_y > abs_x);
#if defined(__CLC_I2F) && (__CLC_GENSIZE_SRC >= __CLC_FPSIZE)
  c = c || __CLC_CONVERT_S_GENTYPE((__CLC_GENTYPE_SRC)__CLC_SRC_MAX == x);
#endif
  __CLC_GENTYPE sel = __clc_select(
      r, __clc_nextafter(r, __clc_sign(r) * (__CLC_GENTYPE)-INFINITY), c);
#if defined(__CLC_I2F) && (__CLC_FPSIZE == 16) && (__CLC_GENSIZE_SRC >= 16)
#if defined(__CLC_GEN_S) && (__CLC_GENSIZE_SRC == 16)
  // short is 16 bits signed, so the maximum value rounded to zero is
  // 0x1.ffcp+14 (0x1p+15 == 32768 > 0x7fff == 32767)
  half dst_max = 0x1.ffcp+14h;
#else
  half dst_max = 0x1.ffcp+15h;
#endif // defined(__CLC_GEN_S) && (__CLC_GENSIZE_SRC == 16)
  half dst_min = -0x1.ffcp+15h;
  sel = __clc_clamp(sel, (__CLC_GENTYPE)dst_min, (__CLC_GENTYPE)dst_max);
#endif // defined(__CLC_I2F) && (__CLC_FPSIZE == 16) &&  (__CLC_GENSIZE_SRC >=
       // 16)
  return sel;
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,
                                                   _rtp)(__CLC_GENTYPE_SRC x) {
  __CLC_GENTYPE r = __CLC_FUNCTION(x);
#ifdef __CLC_I2F
  __CLC_GENTYPE_SRC y =
      __CLC_XCONCAT(__CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC), _sat)(r);
#else
  __CLC_GENTYPE_SRC y = __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC)(r);
#endif
  __CLC_GENTYPE sel =
      __clc_select(r, __clc_nextafter(r, (__CLC_GENTYPE)INFINITY),
                   __CLC_CONVERT_S_GENTYPE(y < x));
#if defined(__CLC_I2F) && (__CLC_FPSIZE == 16) && (__CLC_GENSIZE_SRC >= 16)
  half dst_min = -0x1.ffcp+15h;
  sel = __clc_max(sel, (__CLC_GENTYPE)dst_min);
#endif
  return sel;
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,
                                                   _rtn)(__CLC_GENTYPE_SRC x) {
  __CLC_GENTYPE r = __CLC_FUNCTION(x);
#ifdef __CLC_I2F
  __CLC_GENTYPE_SRC y =
      __CLC_XCONCAT(__CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC), _sat)(r);
#else
  __CLC_GENTYPE_SRC y = __CLC_XCONCAT(__clc_convert_, __CLC_GENTYPE_SRC)(r);
#endif
  __CLC_S_GENTYPE c = __CLC_CONVERT_S_GENTYPE(y > x);
#if defined(__CLC_I2F) && (__CLC_GENSIZE_SRC >= __CLC_FPSIZE)
  c = c || __CLC_CONVERT_S_GENTYPE((__CLC_GENTYPE_SRC)__CLC_SRC_MAX == x);
#endif
  __CLC_GENTYPE sel =
      __clc_select(r, __clc_nextafter(r, (__CLC_GENTYPE)-INFINITY), c);
#if defined(__CLC_I2F) && (__CLC_FPSIZE == 16) && (__CLC_GENSIZE_SRC >= 16)
#if defined(__CLC_GEN_S) && (__CLC_GENSIZE_SRC == 16)
  // short is 16 bits signed, so the maximum value rounded to negative infinity
  // is 0x1.ffcp+14 (0x1p+15 == 32768 > 0x7fff == 32767)
  half dst_max = 0x1.ffcp+14h;
#else
  half dst_max = 0x1.ffcp+15h;
#endif // defined(__CLC_GEN_S) && (__CLC_GENSIZE_SRC == 16)
  sel = __clc_min(sel, (__CLC_GENTYPE)dst_max);
#endif // defined(__CLC_I2F) && (__CLC_FPSIZE == 16) &&
       // (__CLC_GENSIZE_SRC >= 16)
  return sel;
}

#undef __CLC_SRC_MAX
