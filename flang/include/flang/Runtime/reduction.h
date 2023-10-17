//===-- include/flang/Runtime/reduction.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API for the reduction transformational intrinsic functions.

#ifndef FORTRAN_RUNTIME_REDUCTION_H_
#define FORTRAN_RUNTIME_REDUCTION_H_

#include "flang/Common/float128.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"
#include <cfloat>
#include <cinttypes>
#include <complex>
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Reductions that are known to return scalars have per-type entry
// points.  These cover the cases that either have no DIM=
// argument or have an argument rank of 1.  Pass 0 for no DIM=
// or the value of the DIM= argument so that it may be checked.
// The data type in the descriptor is checked against the expected
// return type.
//
// Reductions that return arrays are the remaining cases in which
// the argument rank is greater than one and there is a DIM=
// argument present.  These cases establish and allocate their
// results in a caller-supplied descriptor, which is assumed to
// be large enough.
//
// Complex-valued SUM and PRODUCT reductions and complex-valued
// DOT_PRODUCT have their API entry points defined in complex-reduction.h;
// these here are C wrappers around C++ implementations so as to keep
// usage of C's _Complex types out of C++ code.

// SUM()

std::int8_t RTNAME(SumInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(SumInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(SumInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(SumInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(SumInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTNAME(SumReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(SumReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(SumReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(SumReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if LDBL_MANT_DIG == 64
long double RTNAME(SumReal10)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTNAME(SumReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

void RTNAME(CppSumComplex2)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppSumComplex3)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppSumComplex4)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppSumComplex8)(std::complex<double> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppSumComplex10)(std::complex<long double> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppSumComplex16)(std::complex<long double> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);

void RTNAME(SumDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// PRODUCT()

std::int8_t RTNAME(ProductInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(ProductInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(ProductInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(ProductInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(ProductInteger16)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTNAME(ProductReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(ProductReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(ProductReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(ProductReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if LDBL_MANT_DIG == 64
long double RTNAME(ProductReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTNAME(ProductReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

void RTNAME(CppProductComplex2)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppProductComplex3)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppProductComplex4)(std::complex<float> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppProductComplex8)(std::complex<double> &, const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppProductComplex10)(std::complex<long double> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTNAME(CppProductComplex16)(std::complex<long double> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);

void RTNAME(ProductDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// IALL, IANY, IPARITY
std::int8_t RTNAME(IAll1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(IAll2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(IAll4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(IAll8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(IAll16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTNAME(IAllDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTNAME(IAny1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(IAny2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(IAny4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(IAny8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(IAny16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTNAME(IAnyDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTNAME(IParity1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(IParity2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(IParity4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(IParity8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(IParity16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTNAME(IParityDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// FINDLOC, MAXLOC, & MINLOC
// These return allocated arrays in the supplied descriptor.
// The default value for KIND= should be the default INTEGER in effect at
// compilation time.
void RTNAME(Findloc)(Descriptor &, const Descriptor &x,
    const Descriptor &target, int kind, const char *source, int line,
    const Descriptor *mask = nullptr, bool back = false);
void RTNAME(FindlocDim)(Descriptor &, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const char *source, int line,
    const Descriptor *mask = nullptr, bool back = false);
void RTNAME(MaxlocCharacter)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocInteger1)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocInteger2)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocInteger4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocInteger8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocInteger16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocReal4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocReal8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocReal10)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocReal16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocDim)(Descriptor &, const Descriptor &x, int kind, int dim,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocCharacter)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocInteger1)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocInteger2)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocInteger4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocInteger8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocInteger16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocReal4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocReal8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocReal10)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocReal16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocDim)(Descriptor &, const Descriptor &x, int kind, int dim,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);

// MAXVAL and MINVAL
std::int8_t RTNAME(MaxvalInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(MaxvalInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(MaxvalInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(MaxvalInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(MaxvalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
float RTNAME(MaxvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MaxvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MaxvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(MaxvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if LDBL_MANT_DIG == 64
long double RTNAME(MaxvalReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTNAME(MaxvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTNAME(MaxvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTNAME(MinvalInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(MinvalInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(MinvalInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(MinvalInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(MinvalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
float RTNAME(MinvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MinvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MinvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(MinvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if LDBL_MANT_DIG == 64
long double RTNAME(MinvalReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTNAME(MinvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTNAME(MinvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

void RTNAME(MaxvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);
void RTNAME(MinvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// NORM2
float RTNAME(Norm2_2)(
    const Descriptor &, const char *source, int line, int dim = 0);
float RTNAME(Norm2_3)(
    const Descriptor &, const char *source, int line, int dim = 0);
float RTNAME(Norm2_4)(
    const Descriptor &, const char *source, int line, int dim = 0);
double RTNAME(Norm2_8)(
    const Descriptor &, const char *source, int line, int dim = 0);
#if LDBL_MANT_DIG == 64
long double RTNAME(Norm2_10)(
    const Descriptor &, const char *source, int line, int dim = 0);
#elif LDBL_MANT_DIG == 113
long double RTNAME(Norm2_16)(
    const Descriptor &, const char *source, int line, int dim = 0);
#endif
void RTNAME(Norm2Dim)(
    Descriptor &, const Descriptor &, int dim, const char *source, int line);

// ALL, ANY, COUNT, & PARITY logical reductions
bool RTNAME(All)(const Descriptor &, const char *source, int line, int dim = 0);
void RTNAME(AllDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);
bool RTNAME(Any)(const Descriptor &, const char *source, int line, int dim = 0);
void RTNAME(AnyDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);
std::int64_t RTNAME(Count)(
    const Descriptor &, const char *source, int line, int dim = 0);
void RTNAME(CountDim)(Descriptor &result, const Descriptor &, int dim, int kind,
    const char *source, int line);
bool RTNAME(Parity)(
    const Descriptor &, const char *source, int line, int dim = 0);
void RTNAME(ParityDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);

// DOT_PRODUCT
std::int8_t RTNAME(DotProductInteger1)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int16_t RTNAME(DotProductInteger2)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int32_t RTNAME(DotProductInteger4)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int64_t RTNAME(DotProductInteger8)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#ifdef __SIZEOF_INT128__
common::int128_t RTNAME(DotProductInteger16)(const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
#endif
float RTNAME(DotProductReal2)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
float RTNAME(DotProductReal3)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
float RTNAME(DotProductReal4)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
double RTNAME(DotProductReal8)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#if LDBL_MANT_DIG == 64
long double RTNAME(DotProductReal10)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTNAME(DotProductReal16)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#endif
void RTNAME(CppDotProductComplex2)(std::complex<float> &, const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
void RTNAME(CppDotProductComplex3)(std::complex<float> &, const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
void RTNAME(CppDotProductComplex4)(std::complex<float> &, const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
void RTNAME(CppDotProductComplex8)(std::complex<double> &, const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
#if LDBL_MANT_DIG == 64
void RTNAME(CppDotProductComplex10)(std::complex<long double> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTNAME(CppDotProductComplex16)(std::complex<CppFloat128Type> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
#endif
bool RTNAME(DotProductLogical)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCTION_H_
