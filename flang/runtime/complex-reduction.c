/*===-- flang/runtime/complex-reduction.c ---------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

#include "complex-reduction.h"
#include <float.h>

struct CppComplexFloat {
  float r, i;
};
struct CppComplexDouble {
  double r, i;
};
struct CppComplexLongDouble {
  long double r, i;
};
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
struct CppComplexFloat128 {
  CFloat128Type r, i;
};
#endif

/* Not all environments define CMPLXF, CMPLX, CMPLXL. */

#ifndef CMPLXF
#if defined(__clang_major__) && (__clang_major__ >= 12)
#define CMPLXF __builtin_complex
#else
static float_Complex_t CMPLXF(float r, float i) {
  union {
    struct CppComplexFloat x;
    float_Complex_t result;
  } u;
  u.x.r = r;
  u.x.i = i;
  return u.result;
}
#endif
#endif

#ifndef CMPLX
#if defined(__clang_major__) && (__clang_major__ >= 12)
#define CMPLX __builtin_complex
#else
static double_Complex_t CMPLX(double r, double i) {
  union {
    struct CppComplexDouble x;
    double_Complex_t result;
  } u;
  u.x.r = r;
  u.x.i = i;
  return u.result;
}
#endif
#endif

#ifndef CMPLXL
#if defined(__clang_major__) && (__clang_major__ >= 12)
#define CMPLXL __builtin_complex
#else
static long_double_Complex_t CMPLXL(long double r, long double i) {
  union {
    struct CppComplexLongDouble x;
    long_double_Complex_t result;
  } u;
  u.x.r = r;
  u.x.i = i;
  return u.result;
}
#endif
#endif

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
#ifndef CMPLXF128
/*
 * GCC 7.4.0 (currently minimum GCC version for llvm builds)
 * supports __builtin_complex. For Clang, require >=12.0.
 * Otherwise, rely on the memory layout compatibility.
 */
#if (defined(__clang_major__) && (__clang_major__ >= 12)) || \
    (defined(__GNUC__) && !defined(__clang__))
#define CMPLXF128 __builtin_complex
#else
static CFloat128ComplexType CMPLXF128(CFloat128Type r, CFloat128Type i) {
  union {
    struct CppComplexFloat128 x;
    CFloat128ComplexType result;
  } u;
  u.x.r = r;
  u.x.i = i;
  return u.result;
}
#endif
#endif
#endif

/* RTNAME(SumComplex4) calls RTNAME(CppSumComplex4) with the same arguments
 * and converts the members of its C++ complex result to C _Complex.
 */

#define CPP_NAME(name) Cpp##name
#define ADAPT_REDUCTION(name, cComplex, cpptype, cmplxMacro, ARGS, ARG_NAMES) \
  struct cpptype RTNAME(CPP_NAME(name))(struct cpptype *, ARGS); \
  cComplex RTNAME(name)(ARGS) { \
    struct cpptype result; \
    RTNAME(CPP_NAME(name))(&result, ARG_NAMES); \
    return cmplxMacro(result.r, result.i); \
  }

/* TODO: COMPLEX(2 & 3) */

/* SUM() */
ADAPT_REDUCTION(SumComplex4, float_Complex_t, CppComplexFloat, CMPLXF,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
ADAPT_REDUCTION(SumComplex8, double_Complex_t, CppComplexDouble, CMPLX,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#if LDBL_MANT_DIG == 64
ADAPT_REDUCTION(SumComplex10, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
ADAPT_REDUCTION(SumComplex16, CFloat128ComplexType, CppComplexFloat128,
    CMPLXF128, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif

/* PRODUCT() */
ADAPT_REDUCTION(ProductComplex4, float_Complex_t, CppComplexFloat, CMPLXF,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
ADAPT_REDUCTION(ProductComplex8, double_Complex_t, CppComplexDouble, CMPLX,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#if LDBL_MANT_DIG == 64
ADAPT_REDUCTION(ProductComplex10, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
ADAPT_REDUCTION(ProductComplex16, CFloat128ComplexType, CppComplexFloat128,
    CMPLXF128, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif

/* DOT_PRODUCT() */
ADAPT_REDUCTION(DotProductComplex4, float_Complex_t, CppComplexFloat, CMPLXF,
    DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
ADAPT_REDUCTION(DotProductComplex8, double_Complex_t, CppComplexDouble, CMPLX,
    DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#if LDBL_MANT_DIG == 64
ADAPT_REDUCTION(DotProductComplex10, long_double_Complex_t,
    CppComplexLongDouble, CMPLXL, DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
ADAPT_REDUCTION(DotProductComplex16, CFloat128ComplexType, CppComplexFloat128,
    CMPLXF128, DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#endif

/* REDUCE() */
#define RARGS REDUCE_ARGS(float_Complex_t, float_Complex_t_ref_op)
ADAPT_REDUCTION(ReduceComplex4Ref, float_Complex_t, CppComplexFloat, CMPLXF,
    RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#define RARGS REDUCE_ARGS(float_Complex_t, float_Complex_t_value_op)
ADAPT_REDUCTION(ReduceComplex4Value, float_Complex_t, CppComplexFloat, CMPLXF,
    RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#define RARGS REDUCE_ARGS(double_Complex_t, double_Complex_t_ref_op)
ADAPT_REDUCTION(ReduceComplex8Ref, double_Complex_t, CppComplexDouble, CMPLX,
    RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#define RARGS REDUCE_ARGS(double_Complex_t, double_Complex_t_value_op)
ADAPT_REDUCTION(ReduceComplex8Value, double_Complex_t, CppComplexDouble, CMPLX,
    RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#if LDBL_MANT_DIG == 64
#define RARGS REDUCE_ARGS(long_double_Complex_t, long_double_Complex_t_ref_op)
ADAPT_REDUCTION(ReduceComplex10Ref, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#define RARGS REDUCE_ARGS(long_double_Complex_t, long_double_Complex_t_value_op)
ADAPT_REDUCTION(ReduceComplex10Value, long_double_Complex_t,
    CppComplexLongDouble, CMPLXL, RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
#define RARGS REDUCE_ARGS(CFloat128ComplexType, CFloat128ComplexType_ref_op)
ADAPT_REDUCTION(ReduceComplex16Ref, CFloat128ComplexType, CppComplexFloat128,
    CMPLXF128, RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#define RARGS REDUCE_ARGS(CFloat128ComplexType, CFloat128ComplexType_value_op)
ADAPT_REDUCTION(ReduceComplex16Value, CFloat128ComplexType, CppComplexFloat128,
    CMPLXF128, RARGS, REDUCE_ARG_NAMES)
#undef RARGS
#endif
