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

#include "flang/Common/float128.h"
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
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CFloat128ComplexType RTNAME(SumComplex16)(REDUCTION_ARGS);
#endif

float_Complex_t RTNAME(ProductComplex2)(REDUCTION_ARGS);
float_Complex_t RTNAME(ProductComplex3)(REDUCTION_ARGS);
float_Complex_t RTNAME(ProductComplex4)(REDUCTION_ARGS);
double_Complex_t RTNAME(ProductComplex8)(REDUCTION_ARGS);
long_double_Complex_t RTNAME(ProductComplex10)(REDUCTION_ARGS);
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CFloat128ComplexType RTNAME(ProductComplex16)(REDUCTION_ARGS);
#endif

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
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CFloat128ComplexType RTNAME(DotProductComplex16)(DOT_PRODUCT_ARGS);
#endif

#define REDUCE_ARGS(T) \
  T##_op operation, const struct CppDescriptor *x, \
      const struct CppDescriptor *y, const char *source, int line, \
      int dim /*=0*/, const struct CppDescriptor *mask /*=NULL*/, \
      const T *identity /*=NULL*/, _Bool ordered /*=true*/
#define REDUCE_ARG_NAMES \
  operation, x, y, source, line, dim, mask, identity, ordered

typedef float_Complex_t (*float_Complex_t_op)(
    const float_Complex_t *, const float_Complex_t *);
typedef double_Complex_t (*double_Complex_t_op)(
    const double_Complex_t *, const double_Complex_t *);
typedef long_double_Complex_t (*long_double_Complex_t_op)(
    const long_double_Complex_t *, const long_double_Complex_t *);

float_Complex_t RTNAME(ReduceComplex2)(REDUCE_ARGS(float_Complex_t));
float_Complex_t RTNAME(ReduceComplex3)(REDUCE_ARGS(float_Complex_t));
float_Complex_t RTNAME(ReduceComplex4)(REDUCE_ARGS(float_Complex_t));
double_Complex_t RTNAME(ReduceComplex8)(REDUCE_ARGS(double_Complex_t));
long_double_Complex_t RTNAME(ReduceComplex10)(
    REDUCE_ARGS(long_double_Complex_t));
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
typedef CFloat128ComplexType (*CFloat128ComplexType_op)(
    const CFloat128ComplexType *, const CFloat128ComplexType *);
CFloat128ComplexType RTNAME(ReduceComplex16)(REDUCE_ARGS(CFloat128ComplexType));
#endif

#define REDUCE_DIM_ARGS(T) \
  struct CppDescriptor *result, T##_op operation, \
      const struct CppDescriptor *x, const struct CppDescriptor *y, \
      const char *source, int line, int dim, \
      const struct CppDescriptor *mask /*=NULL*/, const T *identity /*=NULL*/, \
      _Bool ordered /*=true*/
#define REDUCE_DIM_ARG_NAMES \
  result, operation, x, y, source, line, dim, mask, identity, ordered

void RTNAME(ReduceComplex2Dim)(REDUCE_DIM_ARGS(float_Complex_t));
void RTNAME(ReduceComplex3Dim)(REDUCE_DIM_ARGS(float_Complex_t));
void RTNAME(ReduceComplex4Dim)(REDUCE_DIM_ARGS(float_Complex_t));
void RTNAME(ReduceComplex8Dim)(REDUCE_DIM_ARGS(double_Complex_t));
void RTNAME(ReduceComplex10Dim)(REDUCE_DIM_ARGS(long_double_Complex_t));
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTNAME(ReduceComplex16Dim)(REDUCE_DIM_ARGS(CFloat128ComplexType));
#endif

#endif // FORTRAN_RUNTIME_COMPLEX_REDUCTION_H_
