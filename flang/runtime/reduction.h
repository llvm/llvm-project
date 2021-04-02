//===-- runtime/reduction.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the API for the reduction transformational intrinsic functions.
// (Except the complex-valued total reduction forms of SUM and PRODUCT;
// the API for those is in complex-reduction.h so that C's _Complex can
// be used for their return types.)

#ifndef FORTRAN_RUNTIME_REDUCTION_H_
#define FORTRAN_RUNTIME_REDUCTION_H_

#include "descriptor.h"
#include "entry-names.h"
#include "flang/Common/uint128.h"
#include <complex>
#include <cstdint>

namespace Fortran::runtime {
extern "C" {

// Reductions that are known to return scalars have per-type entry
// points.  These cover the casse that either have no DIM=
// argument, or have an argument rank of 1.  Pass 0 for no DIM=
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
// Complex-valued SUM and PRODUCT reductions have their API
// entry points defined in complex-reduction.h; these are C wrappers
// around C++ implementations so as to keep usage of C's _Complex
// types out of C++ code.

// SUM()

std::int8_t RTNAME(SumInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(SumInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(SumInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(SumInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
common::int128_t RTNAME(SumInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTNAME(SumReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(SumReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(SumReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(SumReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(SumReal10)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(SumReal16)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);

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
common::int128_t RTNAME(ProductInteger16)(const Descriptor &,
    const char *source, int line, int dim = 0,
    const Descriptor *mask = nullptr);

// REAL/COMPLEX(2 & 3) return 32-bit float results for the caller to downconvert
float RTNAME(ProductReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(ProductReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(ProductReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(ProductReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(ProductReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(ProductReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);

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

// MAXLOC and MINLOC
// These return allocated arrays in the supplied descriptor.
// The default value for KIND= should be the default INTEGER in effect at
// compilation time.
void RTNAME(Maxloc)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MaxlocDim)(Descriptor &, const Descriptor &, int kind, int dim,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(Minloc)(Descriptor &, const Descriptor &, int kind,
    const char *source, int line, const Descriptor *mask = nullptr,
    bool back = false);
void RTNAME(MinlocDim)(Descriptor &, const Descriptor &, int kind, int dim,
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
common::int128_t RTNAME(MaxvalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MaxvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MaxvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MaxvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(MaxvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(MaxvalReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(MaxvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
void RTNAME(MaxvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

std::int8_t RTNAME(MinvalInteger1)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int16_t RTNAME(MinvalInteger2)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int32_t RTNAME(MinvalInteger4)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
std::int64_t RTNAME(MivalInteger8)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
common::int128_t RTNAME(MivalInteger16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MinvalReal2)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MinvalReal3)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
float RTNAME(MinvalReal4)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
double RTNAME(MinvalReal8)(const Descriptor &, const char *source, int line,
    int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(MinvalReal10)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
long double RTNAME(MinvalReal16)(const Descriptor &, const char *source,
    int line, int dim = 0, const Descriptor *mask = nullptr);
void RTNAME(MinvalCharacter)(Descriptor &, const Descriptor &,
    const char *source, int line, const Descriptor *mask = nullptr);

void RTNAME(MaxvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);
void RTNAME(MinvalDim)(Descriptor &, const Descriptor &, int dim,
    const char *source, int line, const Descriptor *mask = nullptr);

// ALL, ANY, & COUNT logical reductions
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

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_REDUCTION_H_
