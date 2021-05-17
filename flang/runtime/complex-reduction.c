/*===-- flang/runtime/complex-reduction.c ---------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

#include "complex-reduction.h"
#include "flang/Common/long-double.h"

struct CppComplexFloat {
  float r, i;
};
struct CppComplexDouble {
  double r, i;
};
struct CppComplexLongDouble {
  long double r, i;
};

/* Not all environments define CMPLXF, CMPLX, CMPLXL. */

#ifndef CMPLXF
#if __clang_major__ >= 12
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
#if __clang_major__ >= 12
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
#if __clang_major__ >= 12
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
#if LONG_DOUBLE == 80
ADAPT_REDUCTION(SumComplex10, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#elif LONG_DOUBLE == 128
ADAPT_REDUCTION(SumComplex16, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif

/* PRODUCT() */
ADAPT_REDUCTION(ProductComplex4, float_Complex_t, CppComplexFloat, CMPLXF,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
ADAPT_REDUCTION(ProductComplex8, double_Complex_t, CppComplexDouble, CMPLX,
    REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#if LONG_DOUBLE == 80
ADAPT_REDUCTION(ProductComplex10, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#elif LONG_DOUBLE == 128
ADAPT_REDUCTION(ProductComplex16, long_double_Complex_t, CppComplexLongDouble,
    CMPLXL, REDUCTION_ARGS, REDUCTION_ARG_NAMES)
#endif

/* DOT_PRODUCT() */
ADAPT_REDUCTION(DotProductComplex4, float_Complex_t, CppComplexFloat, CMPLXF,
    DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
ADAPT_REDUCTION(DotProductComplex8, double_Complex_t, CppComplexDouble, CMPLX,
    DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#if LONG_DOUBLE == 80
ADAPT_REDUCTION(DotProductComplex10, long_double_Complex_t,
    CppComplexLongDouble, CMPLXL, DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#elif LONG_DOUBLE == 128
ADAPT_REDUCTION(DotProductComplex16, long_double_Complex_t,
    CppComplexLongDouble, CMPLXL, DOT_PRODUCT_ARGS, DOT_PRODUCT_ARG_NAMES)
#endif
