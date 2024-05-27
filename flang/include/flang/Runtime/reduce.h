//===-- include/flang/Runtime/reduce.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API for implementations of the transformational intrinsic
// function REDUCE(); see F'2023 16.9.173.
//
// Similar to the definition of the APIs for SUM(), &c., in reduction.h,
// there are typed functions here like ReduceInteger4() for total reductions
// to scalars and void functions like ReduceInteger4Dim() for partial
// reductions to smaller arrays.

#ifndef FORTRAN_RUNTIME_REDUCE_H_
#define FORTRAN_RUNTIME_REDUCE_H_

#include "flang/Common/float128.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"
#include <complex>
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

template <typename T> using ReductionOperation = T (*)(const T *, const T *);
template <typename CHAR>
using ReductionCharOperation = void (*)(CHAR *hiddenResult,
    std::size_t resultLen, const CHAR *x, const CHAR *y, std::size_t xLen,
    std::size_t yLen);
using ReductionDerivedTypeOperation = void (*)(
    void *hiddenResult, const void *x, const void *y);

extern "C" {

std::int8_t RTDECL(ReduceInteger1)(const Descriptor &,
    ReductionOperation<std::int8_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int8_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceInteger1Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int8_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int8_t *identity = nullptr,
    bool ordered = true);
std::int16_t RTDECL(ReduceInteger2)(const Descriptor &,
    ReductionOperation<std::int16_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int16_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceInteger2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int16_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int16_t *identity = nullptr,
    bool ordered = true);
std::int32_t RTDECL(ReduceInteger4)(const Descriptor &,
    ReductionOperation<std::int32_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int32_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceInteger4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int32_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int32_t *identity = nullptr,
    bool ordered = true);
std::int64_t RTDECL(ReduceInteger8)(const Descriptor &,
    ReductionOperation<std::int64_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int64_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceInteger8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int64_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int64_t *identity = nullptr,
    bool ordered = true);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(ReduceInteger16)(const Descriptor &,
    ReductionOperation<common::int128_t>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const common::int128_t *identity = nullptr, bool ordered = true);
void RTDECL(ReduceInteger16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<common::int128_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr,
    const common::int128_t *identity = nullptr, bool ordered = true);
#endif

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTDECL(ReduceReal2)(const Descriptor &, ReductionOperation<float>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const float *identity = nullptr, bool ordered = true);
void RTDECL(ReduceReal2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<float>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const float *identity = nullptr,
    bool ordered = true);
float RTDECL(ReduceReal3)(const Descriptor &, ReductionOperation<float>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const float *identity = nullptr, bool ordered = true);
void RTDECL(ReduceReal3Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<float>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const float *identity = nullptr,
    bool ordered = true);
float RTDECL(ReduceReal4)(const Descriptor &, ReductionOperation<float>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const float *identity = nullptr, bool ordered = true);
void RTDECL(ReduceReal4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<float>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const float *identity = nullptr,
    bool ordered = true);
double RTDECL(ReduceReal8)(const Descriptor &, ReductionOperation<double>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const double *identity = nullptr, bool ordered = true);
void RTDECL(ReduceReal8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<double>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const double *identity = nullptr,
    bool ordered = true);
#if LDBL_MANT_DIG == 64
long double RTDECL(ReduceReal10)(const Descriptor &,
    ReductionOperation<long double>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const long double *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceReal10Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<long double>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const long double *identity = nullptr,
    bool ordered = true);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTDECL(ReduceReal16)(const Descriptor &,
    ReductionOperation<CppFloat128Type>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const CppFloat128Type *identity = nullptr, bool ordered = true);
void RTDECL(ReduceReal16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<CppFloat128Type>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const CppFloat128Type *identity = nullptr,
    bool ordered = true);
#endif

void RTDECL(CppReduceComplex2)(std::complex<float> &, const Descriptor &,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex3)(std::complex<float> &, const Descriptor &,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex3Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex4)(std::complex<float> &, const Descriptor &,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<float>>, const char *source, int line,
    int dim, const Descriptor *mask = nullptr,
    const std::complex<float> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex8)(std::complex<double> &, const Descriptor &,
    ReductionOperation<std::complex<double>>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<double> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<double>>, const char *source, int line,
    int dim, const Descriptor *mask = nullptr,
    const std::complex<double> *identity = nullptr, bool ordered = true);
#if LDBL_MANT_DIG == 64
void RTDECL(CppReduceComplex10)(std::complex<long double> &, const Descriptor &,
    ReductionOperation<std::complex<long double>>, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<long double> *identity = nullptr, bool ordered = true);
void RTDECL(CppReduceComplex10Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<long double>>, const char *source, int line,
    int dim, const Descriptor *mask = nullptr,
    const std::complex<long double> *identity = nullptr, bool ordered = true);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTDECL(CppReduceComplex16)(std::complex<CppFloat128Type> &,
    const Descriptor &, ReductionOperation<std::complex<CppFloat128Type>>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const std::complex<CppFloat128Type> *identity = nullptr,
    bool ordered = true);
void RTDECL(CppReduceComplex16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<CppFloat128Type>>, const char *source,
    int line, int dim, const Descriptor *mask = nullptr,
    const std::complex<CppFloat128Type> *identity = nullptr,
    bool ordered = true);
#endif

bool RTDECL(ReduceLogical1)(const Descriptor &, ReductionOperation<std::int8_t>,
    const char *source, int line, int dim = 0, const Descriptor *mask = nullptr,
    const std::int8_t *identity = nullptr, bool ordered = true);
void RTDECL(ReduceLogical1Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int8_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int8_t *identity = nullptr,
    bool ordered = true);
bool RTDECL(ReduceLogical2)(const Descriptor &,
    ReductionOperation<std::int16_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int16_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceLogical2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int16_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int16_t *identity = nullptr,
    bool ordered = true);
bool RTDECL(ReduceLogical4)(const Descriptor &,
    ReductionOperation<std::int32_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int32_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceLogical4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int32_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int32_t *identity = nullptr,
    bool ordered = true);
bool RTDECL(ReduceLogical8)(const Descriptor &,
    ReductionOperation<std::int64_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const std::int64_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceLogical8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int64_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const std::int64_t *identity = nullptr,
    bool ordered = true);

void RTDECL(ReduceChar1)(char *result, const Descriptor &array,
    ReductionCharOperation<char>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const char *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceCharacter1Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const char *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceChar2)(char16_t *result, const Descriptor &array,
    ReductionCharOperation<char16_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const char16_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceCharacter2Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char16_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const char16_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceChar4)(char32_t *result, const Descriptor &array,
    ReductionCharOperation<char32_t>, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const char32_t *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceCharacter4Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char32_t>, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const char32_t *identity = nullptr,
    bool ordered = true);

void RTDECL(ReduceDerivedType)(char *result, const Descriptor &array,
    ReductionDerivedTypeOperation, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr, const char *identity = nullptr,
    bool ordered = true);
void RTDECL(ReduceDerivedTypeDim)(Descriptor &result, const Descriptor &array,
    ReductionDerivedTypeOperation, const char *source, int line, int dim,
    const Descriptor *mask = nullptr, const char *identity = nullptr,
    bool ordered = true);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCE_H_
