/*===-- flang/runtime/complex-reduction.h ---------------------------*- C -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

/* Wraps the C++-coded complex-valued SUM and PRODUCT reductions with
 * C-coded wrapper functions returning _Complex values, to avoid problems
 * with C++ build compilers that don't support C's _Complex.
 */

#ifndef FORTRAN_RUNTIME_COMPLEX_REDUCTION_H_
#define FORTRAN_RUNTIME_COMPLEX_REDUCTION_H_

#include "flang/Runtime/entry-names.h"
#include <complex.h>

struct CppDescriptor; /* dummy type name for Fortran::runtime::Descriptor */

#if defined(_MSC_VER) && !(defined(__clang_major__) && __clang_major__ >= 12)
typedef _Fcomplex float_Complex_t;
typedef _Dcomplex double_Complex_t;
typedef _Lcomplex long_double_Complex_t;
#else
typedef float _Complex float_Complex_t;
typedef double _Complex double_Complex_t;
typedef long double _Complex long_double_Complex_t;
#endif

#define REDUCTION_ARGS \
  const struct CppDescriptor *x, const char *source, int line, int dim /*=0*/, \
      const struct CppDescriptor *mask /*=NULL*/
#define REDUCTION_ARG_NAMES x, source, line, dim, mask

float_Complex_t RTNAME(SumComplex2)(REDUCTION_ARGS);
float_Complex_t RTNAME(SumComplex3)(REDUCTION_ARGS);
float_Complex_t RTNAME(SumComplex4)(REDUCTION_ARGS);
double_Complex_t RTNAME(SumComplex8)(REDUCTION_ARGS);
long_double_Complex_t RTNAME(SumComplex10)(REDUCTION_ARGS);
long_double_Complex_t RTNAME(SumComplex16)(REDUCTION_ARGS);

float_Complex_t RTNAME(ProductComplex2)(REDUCTION_ARGS);
float_Complex_t RTNAME(ProductComplex3)(REDUCTION_ARGS);
float_Complex_t RTNAME(ProductComplex4)(REDUCTION_ARGS);
double_Complex_t RTNAME(ProductComplex8)(REDUCTION_ARGS);
long_double_Complex_t RTNAME(ProductComplex10)(REDUCTION_ARGS);
long_double_Complex_t RTNAME(ProductComplex16)(REDUCTION_ARGS);

#define DOT_PRODUCT_ARGS \
  const struct CppDescriptor *x, const struct CppDescriptor *y, \
      const char *source, int line, int dim /*=0*/, \
      const struct CppDescriptor *mask /*=NULL*/
#define DOT_PRODUCT_ARG_NAMES x, y, source, line, dim, mask

float_Complex_t RTNAME(DotProductComplex2)(DOT_PRODUCT_ARGS);
float_Complex_t RTNAME(DotProductComplex3)(DOT_PRODUCT_ARGS);
float_Complex_t RTNAME(DotProductComplex4)(DOT_PRODUCT_ARGS);
double_Complex_t RTNAME(DotProductComplex8)(DOT_PRODUCT_ARGS);
long_double_Complex_t RTNAME(DotProductComplex10)(DOT_PRODUCT_ARGS);
long_double_Complex_t RTNAME(DotProductComplex16)(DOT_PRODUCT_ARGS);

#endif // FORTRAN_RUNTIME_COMPLEX_REDUCTION_H_
