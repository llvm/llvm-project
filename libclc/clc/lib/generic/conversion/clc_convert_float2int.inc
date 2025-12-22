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
// Casting from floating point to integer results in conversions
// with truncation, so it should be suitable for the default convert
// functions.
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

#ifdef __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_FUNCTION_S(__CLC_GENTYPE x) {
  return (__CLC_S_GENTYPE_DST)x;
}
_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_FUNCTION_U(__CLC_GENTYPE x) {
  return (__CLC_U_GENTYPE_DST)x;
}

#else // __CLC_SCALAR

_CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_FUNCTION_S(__CLC_GENTYPE x) {
  return __builtin_convertvector(x, __CLC_S_GENTYPE_DST);
}
_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_FUNCTION_U(__CLC_GENTYPE x) {
  return __builtin_convertvector(x, __CLC_U_GENTYPE_DST);
}

#endif // __CLC_SCALAR

// Saturated conversion

#if __CLC_GENSIZE_DST == 8
#define __CLC_DST_MAX_S SCHAR_MAX
#define __CLC_DST_MIN_S SCHAR_MIN
#define __CLC_DST_MAX_U UCHAR_MAX
#define __CLC_DST_MIN_U UCHAR_MIN
#elif __CLC_GENSIZE_DST == 16
#define __CLC_DST_MAX_S SHRT_MAX
#define __CLC_DST_MIN_S SHRT_MIN
#define __CLC_DST_MAX_U USHRT_MAX
#define __CLC_DST_MIN_U USHRT_MIN
#elif __CLC_GENSIZE_DST == 32
#define __CLC_DST_MAX_S INT_MAX
#define __CLC_DST_MIN_S INT_MIN
#define __CLC_DST_MAX_U UINT_MAX
#define __CLC_DST_MIN_U UINT_MIN
#elif __CLC_GENSIZE_DST == 64
#define __CLC_DST_MAX_S LONG_MAX
#define __CLC_DST_MIN_S LONG_MIN
#define __CLC_DST_MAX_U ULONG_MAX
#define __CLC_DST_MIN_U ULONG_MIN
#endif

_CLC_OVERLOAD
_CLC_DEF __CLC_S_GENTYPE_DST __CLC_FUNCTION_S_SAT(__CLC_GENTYPE x) {
  __CLC_S_GENTYPE_DST y = __CLC_FUNCTION_S(x);
  y = __clc_select(y, (__CLC_S_GENTYPE_DST)__CLC_DST_MIN_S,
                   __CLC_FUNCTION_S(x <= (__CLC_GENTYPE)__CLC_DST_MIN_S));
  y = __clc_select(y, (__CLC_S_GENTYPE_DST)__CLC_DST_MAX_S,
                   __CLC_FUNCTION_S(x >= (__CLC_GENTYPE)__CLC_DST_MAX_S));
  return y;
}
_CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST
__CLC_FUNCTION_U_SAT(__CLC_GENTYPE x) {
  __CLC_U_GENTYPE_DST y = __CLC_FUNCTION_U(x);
  // FIXME: This is a work around for lack of select function with signed
  // third argument when the first two arguments are unsigned types. We cast
  // to the signed type for sign-extension, then do a bitcast to the unsigned
  // type.
  y = __clc_select(y, (__CLC_U_GENTYPE_DST)__CLC_DST_MIN_U,
                   __CLC_XCONCAT(__clc_as_, __CLC_U_GENTYPE_DST)(
                       __CLC_FUNCTION_S(x <= (__CLC_GENTYPE)__CLC_DST_MIN_U)));
  y = __clc_select(y, (__CLC_U_GENTYPE_DST)__CLC_DST_MAX_U,
                   __CLC_XCONCAT(__clc_as_, __CLC_U_GENTYPE_DST)(
                       __CLC_FUNCTION_S(x >= (__CLC_GENTYPE)__CLC_DST_MAX_U)));
  return y;
}

#undef __CLC_DST_MAX_S
#undef __CLC_DST_MIN_S
#undef __CLC_DST_MAX_U
#undef __CLC_DST_MIN_U

// Saturated conversion with rounding mode

#define __CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE(MODE, ROUND_FUNC)              \
  _CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_S_SAT, MODE)(__CLC_GENTYPE x) {                           \
    x = ROUND_FUNC(x);                                                         \
    return __CLC_FUNCTION_S_SAT(x);                                            \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_U_SAT, MODE)(__CLC_GENTYPE x) {                           \
    x = ROUND_FUNC(x);                                                         \
    return __CLC_FUNCTION_U_SAT(x);                                            \
  }

__CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE(_rte, __clc_rint)
__CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE(_rtz, )
__CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE(_rtp, __clc_ceil)
__CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE(_rtn, __clc_floor)

#undef __CLC_DEFINE_FUNCTION_SAT_ROUNDING_MODE

// Rounding mode conversion

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE, ROUND_FUNC)                  \
  _CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_S, MODE)(__CLC_GENTYPE x) {                               \
    x = ROUND_FUNC(x);                                                         \
    return __CLC_FUNCTION_S(x);                                                \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_U, MODE)(__CLC_GENTYPE x) {                               \
    x = ROUND_FUNC(x);                                                         \
    return __CLC_FUNCTION_U(x);                                                \
  }

__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rte, __clc_rint)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtz, )
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtp, __clc_ceil)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtn, __clc_floor)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE
