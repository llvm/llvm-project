//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Default conversion

// All conversions are in accordance with the OpenCL specification,
// which cites the C99 conversion rules.
//
// While C99 allows dynamically changing the rounding mode
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

#define __CLC_GENSIZE_DST __CLC_GENSIZE

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_S_GENTYPE_SRC x) {
#ifdef __CLC_SCALAR
  return x;
#else
  return __builtin_convertvector(x, __CLC_GENTYPE);
#endif
}
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION(__CLC_U_GENTYPE_SRC x) {
#ifdef __CLC_SCALAR
  return x;
#else
  return __builtin_convertvector(x, __CLC_GENTYPE);
#endif
}

#ifdef __CLC_GEN_S

#if __CLC_GENSIZE == 8
#define __CLC_DST_MAX SCHAR_MAX
#define __CLC_DST_MIN SCHAR_MIN
#elif __CLC_GENSIZE == 16
#define __CLC_DST_MAX SHRT_MAX
#define __CLC_DST_MIN SHRT_MIN
#elif __CLC_GENSIZE == 32
#define __CLC_DST_MAX INT_MAX
#define __CLC_DST_MIN INT_MIN
#elif __CLC_GENSIZE == 64
#define __CLC_DST_MAX LONG_MAX
#define __CLC_DST_MIN LONG_MIN
#endif

#else // __CLC_GEN_S

#if __CLC_GENSIZE == 8
#define __CLC_DST_MAX UCHAR_MAX
#define __CLC_DST_MIN UCHAR_MIN
#elif __CLC_GENSIZE == 16
#define __CLC_DST_MAX USHRT_MAX
#define __CLC_DST_MIN USHRT_MIN
#elif __CLC_GENSIZE == 32
#define __CLC_DST_MAX UINT_MAX
#define __CLC_DST_MIN UINT_MIN
#elif __CLC_GENSIZE == 64
#define __CLC_DST_MAX ULONG_MAX
#define __CLC_DST_MIN ULONG_MIN
#endif

#endif // __CLC_GEN_S

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_SAT(__CLC_S_GENTYPE_SRC x) {
#if defined(__CLC_GEN_S) && (__CLC_GENSIZE_SRC == __CLC_GENSIZE_DST)
  // Conversion between same types
  return x;
#else
#if __CLC_GENSIZE_SRC == __CLC_GENSIZE_DST
  x = __clc_max(x, (__CLC_S_SCALAR_TYPE_SRC)0);
#elif __CLC_GENSIZE_SRC > __CLC_GENSIZE_DST
  x = __clc_clamp(x, (__CLC_S_SCALAR_TYPE_SRC)__CLC_DST_MIN,
                  (__CLC_S_SCALAR_TYPE_SRC)__CLC_DST_MAX);
#elif defined(__CLC_GEN_U)
  // sizeof(src) < sizeof(dst)
  x = __clc_max(x, (__CLC_S_SCALAR_TYPE_SRC)0);
#endif
  return __CLC_FUNCTION(x);
#endif
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNCTION_SAT(__CLC_U_GENTYPE_SRC x) {
#if defined(__CLC_GEN_U) && (__CLC_GENSIZE_SRC == __CLC_GENSIZE_DST)
  // Conversion between same types
  return x;
#else
#if __CLC_GENSIZE_SRC >= __CLC_GENSIZE_DST
  x = __clc_min(x, (__CLC_U_SCALAR_TYPE_SRC)__CLC_DST_MAX);
#endif
  return __CLC_FUNCTION(x);
#endif
}

#ifdef __CLC_SCALAR
#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_NONSAT_IMPL(TYPE_SRC, MODE)        \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,           \
                                                     MODE)(TYPE_SRC x) {       \
    return x;                                                                  \
  }
#else
#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_NONSAT_IMPL(TYPE_SRC, MODE)        \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,           \
                                                     MODE)(TYPE_SRC x) {       \
    return __builtin_convertvector(x, __CLC_GENTYPE);                          \
  }
#endif

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(TYPE_SRC, MODE)               \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_NONSAT_IMPL(TYPE_SRC, MODE)              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION_SAT,       \
                                                     MODE)(TYPE_SRC x) {       \
    return __CLC_FUNCTION_SAT(x);                                              \
  }

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(__CLC_S_GENTYPE_SRC, MODE)          \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(__CLC_U_GENTYPE_SRC, MODE)

__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rte)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtz)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtp)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtn)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE_NONSAT_IMPL
#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL
#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE

#undef __CLC_DST_MAX
#undef __CLC_DST_MIN

#undef __CLC_GENSIZE_DST
