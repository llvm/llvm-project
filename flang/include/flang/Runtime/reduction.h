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

std::int8_t RTDECL(SumInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(SumInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(SumInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(SumInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(SumInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTDECL(SumReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(SumReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(SumReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTDECL(SumReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(SumReal10)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(SumReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

void RTDECL(CppSumComplex2)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppSumComplex3)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppSumComplex4)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppSumComplex8)(CppTypeFor<TypeCategory::Complex, 8> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#if HAS_FLOAT80
void RTDECL(CppSumComplex10)(CppTypeFor<TypeCategory::Complex, 10> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDECL(CppSumComplex16)(CppTypeFor<TypeCategory::Complex, 16> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif

void RTDECL(SumDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// PRODUCT()

std::int8_t RTDECL(ProductInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(ProductInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(ProductInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(ProductInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(ProductInteger16)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTDECL(ProductReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(ProductReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(ProductReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTDECL(ProductReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(ProductReal10)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(ProductReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif

void RTDECL(CppProductComplex2)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppProductComplex3)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
void RTDECL(CppProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#if HAS_FLOAT80
void RTDECL(CppProductComplex10)(CppTypeFor<TypeCategory::Complex, 10> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDECL(CppProductComplex16)(CppTypeFor<TypeCategory::Complex, 16> &,
    const Descriptor &, const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif

void RTDECL(ProductDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// IALL, IANY, IPARITY
std::int8_t RTDECL(IAll1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(IAll2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(IAll4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(IAll8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(IAll16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTDECL(IAllDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTDECL(IAny1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(IAny2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(IAny4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(IAny8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(IAny16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTDECL(IAnyDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTDECL(IParity1)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(IParity2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(IParity4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(IParity8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(IParity16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTDECL(IParityDim)(Descriptor &result, const Descriptor &array, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// FINDLOC, MAXLOC, & MINLOC
// These return allocated arrays in the supplied descriptor.
// The default value for KIND= should be the default INTEGER in effect at
// compilation time.
void RTDECL(Findloc)(Descriptor &, const Descriptor &x,
    const Descriptor &target, int kind, const char *source, int line,
    const Descriptor *mask = nullptr, bool back = false);
void RTDECL(FindlocDim)(Descriptor &, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const char *source, int line,
    const Descriptor *mask = nullptr, bool back = false);
void RTDECL(MaxlocCharacter)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocInteger1)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocInteger2)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocInteger4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocInteger8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocInteger16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocReal4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocReal8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocReal10)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocReal16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MaxlocDim)(Descriptor &, const Descriptor &x, int kind, int dim,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocCharacter)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocInteger1)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocInteger2)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocInteger4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocInteger8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocInteger16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocReal4)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocReal8)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocReal10)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocReal16)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTDECL(MinlocDim)(Descriptor &, const Descriptor &x, int kind, int dim,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);

// MAXVAL and MINVAL
std::int8_t RTDECL(MaxvalInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(MaxvalInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(MaxvalInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(MaxvalInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(MaxvalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
float RTDECL(MaxvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(MaxvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(MaxvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTDECL(MaxvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(MaxvalReal10)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(MaxvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTDECL(MaxvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTDECL(MinvalInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTDECL(MinvalInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTDECL(MinvalInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTDECL(MinvalInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(MinvalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
float RTDECL(MinvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(MinvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTDECL(MinvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTDECL(MinvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(MinvalReal10)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(MinvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
#endif
void RTDECL(MinvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

void RTDECL(MaxvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);
void RTDECL(MinvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// NORM2
float RTDECL(Norm2_2)(
    const Descriptor &, const char *source, int line, int dim = 0);
float RTDECL(Norm2_3)(
    const Descriptor &, const char *source, int line, int dim = 0);
float RTDECL(Norm2_4)(
    const Descriptor &, const char *source, int line, int dim = 0);
double RTDECL(Norm2_8)(
    const Descriptor &, const char *source, int line, int dim = 0);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(Norm2_10)(
    const Descriptor &, const char *source, int line, int dim = 0);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(Norm2_16)(
    const Descriptor &, const char *source, int line, int dim = 0);
void RTDECL(Norm2DimReal16)(
    Descriptor &, const Descriptor &, int dim, const char *source, int line);
#endif
void RTDECL(Norm2Dim)(
    Descriptor &, const Descriptor &, int dim, const char *source, int line);

// ALL, ANY, COUNT, & PARITY logical reductions
bool RTDECL(All)(const Descriptor &, const char *source, int line, int dim = 0);
void RTDECL(AllDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);
bool RTDECL(Any)(const Descriptor &, const char *source, int line, int dim = 0);
void RTDECL(AnyDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);
std::int64_t RTDECL(Count)(
    const Descriptor &, const char *source, int line, int dim = 0);
void RTDECL(CountDim)(Descriptor &result, const Descriptor &, int dim, int kind,
    const char *source, int line);
bool RTDECL(Parity)(
    const Descriptor &, const char *source, int line, int dim = 0);
void RTDECL(ParityDim)(Descriptor &result, const Descriptor &, int dim,
    const char *source, int line);

// DOT_PRODUCT
std::int8_t RTDECL(DotProductInteger1)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int16_t RTDECL(DotProductInteger2)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int32_t RTDECL(DotProductInteger4)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
std::int64_t RTDECL(DotProductInteger8)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#ifdef __SIZEOF_INT128__
common::int128_t RTDECL(DotProductInteger16)(const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
#endif
float RTDECL(DotProductReal2)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
float RTDECL(DotProductReal3)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
float RTDECL(DotProductReal4)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
double RTDECL(DotProductReal8)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDECL(DotProductReal10)(const Descriptor &,
    const Descriptor &, const char *source = nullptr, int line = 0);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDECL(DotProductReal16)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);
#endif
void RTDECL(CppDotProductComplex2)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
void RTDECL(CppDotProductComplex3)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
void RTDECL(CppDotProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
void RTDECL(CppDotProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
#if HAS_FLOAT80
void RTDECL(CppDotProductComplex10)(CppTypeFor<TypeCategory::Complex, 10> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDECL(CppDotProductComplex16)(CppTypeFor<TypeCategory::Complex, 16> &,
    const Descriptor &, const Descriptor &, const char *source = nullptr,
    int line = 0);
#endif
bool RTDECL(DotProductLogical)(const Descriptor &, const Descriptor &,
    const char *source = nullptr, int line = 0);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCTION_H_
