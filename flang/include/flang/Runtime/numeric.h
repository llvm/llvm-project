//===-- include/flang/Runtime/numeric.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines API between compiled code and the implementations of various numeric
// intrinsic functions in the runtime library.

#ifndef FORTRAN_RUNTIME_NUMERIC_H_
#define FORTRAN_RUNTIME_NUMERIC_H_

#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
extern "C" {

// CEILING
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// EXPONENT is defined to return default INTEGER; support INTEGER(4 & 8)
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// FLOOR
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// FRACTION
CppTypeFor<TypeCategory::Real, 4> RTNAME(Fraction4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTNAME(Fraction8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Fraction10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Fraction16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// ISNAN / IEEE_IS_NAN
bool RTNAME(IsNaN4)(CppTypeFor<TypeCategory::Real, 4>);
bool RTNAME(IsNaN8)(CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
bool RTNAME(IsNaN10)(CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
bool RTNAME(IsNaN16)(CppTypeFor<TypeCategory::Real, 16>);
#endif

// MOD & MODULO
CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModInteger1)(
    CppTypeFor<TypeCategory::Integer, 1>, CppTypeFor<TypeCategory::Integer, 1>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModInteger2)(
    CppTypeFor<TypeCategory::Integer, 2>, CppTypeFor<TypeCategory::Integer, 2>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModInteger4)(
    CppTypeFor<TypeCategory::Integer, 4>, CppTypeFor<TypeCategory::Integer, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModInteger8)(
    CppTypeFor<TypeCategory::Integer, 8>, CppTypeFor<TypeCategory::Integer, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModInteger16)(
    CppTypeFor<TypeCategory::Integer, 16>,
    CppTypeFor<TypeCategory::Integer, 16>, const char *sourceFile = nullptr,
    int sourceLine = 0);
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModReal4)(
    CppTypeFor<TypeCategory::Real, 4>, CppTypeFor<TypeCategory::Real, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModReal8)(
    CppTypeFor<TypeCategory::Real, 8>, CppTypeFor<TypeCategory::Real, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModReal10)(
    CppTypeFor<TypeCategory::Real, 10>, CppTypeFor<TypeCategory::Real, 10>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModReal16)(
    CppTypeFor<TypeCategory::Real, 16>, CppTypeFor<TypeCategory::Real, 16>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModuloInteger1)(
    CppTypeFor<TypeCategory::Integer, 1>, CppTypeFor<TypeCategory::Integer, 1>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModuloInteger2)(
    CppTypeFor<TypeCategory::Integer, 2>, CppTypeFor<TypeCategory::Integer, 2>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModuloInteger4)(
    CppTypeFor<TypeCategory::Integer, 4>, CppTypeFor<TypeCategory::Integer, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModuloInteger8)(
    CppTypeFor<TypeCategory::Integer, 8>, CppTypeFor<TypeCategory::Integer, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModuloInteger16)(
    CppTypeFor<TypeCategory::Integer, 16>,
    CppTypeFor<TypeCategory::Integer, 16>, const char *sourceFile = nullptr,
    int sourceLine = 0);
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModuloReal4)(
    CppTypeFor<TypeCategory::Real, 4>, CppTypeFor<TypeCategory::Real, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModuloReal8)(
    CppTypeFor<TypeCategory::Real, 8>, CppTypeFor<TypeCategory::Real, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModuloReal10)(
    CppTypeFor<TypeCategory::Real, 10>, CppTypeFor<TypeCategory::Real, 10>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModuloReal16)(
    CppTypeFor<TypeCategory::Real, 16>, CppTypeFor<TypeCategory::Real, 16>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif

// NINT
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#if defined __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// NEAREST
// The second argument to NEAREST is the result of a comparison
// to zero (i.e., S > 0)
CppTypeFor<TypeCategory::Real, 4> RTNAME(Nearest4)(
    CppTypeFor<TypeCategory::Real, 4>, bool positive);
CppTypeFor<TypeCategory::Real, 8> RTNAME(Nearest8)(
    CppTypeFor<TypeCategory::Real, 8>, bool positive);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Nearest10)(
    CppTypeFor<TypeCategory::Real, 10>, bool positive);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Nearest16)(
    CppTypeFor<TypeCategory::Real, 16>, bool positive);
#endif

// RRSPACING
CppTypeFor<TypeCategory::Real, 4> RTNAME(RRSpacing4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTNAME(RRSpacing8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(RRSpacing10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(RRSpacing16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// SET_EXPONENT's I= argument can be any INTEGER kind; upcast it to 64-bit
CppTypeFor<TypeCategory::Real, 4> RTNAME(SetExponent4)(
    CppTypeFor<TypeCategory::Real, 4>, std::int64_t);
CppTypeFor<TypeCategory::Real, 8> RTNAME(SetExponent8)(
    CppTypeFor<TypeCategory::Real, 8>, std::int64_t);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(SetExponent10)(
    CppTypeFor<TypeCategory::Real, 10>, std::int64_t);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(SetExponent16)(
    CppTypeFor<TypeCategory::Real, 16>, std::int64_t);
#endif

// SCALE
CppTypeFor<TypeCategory::Real, 4> RTNAME(Scale4)(
    CppTypeFor<TypeCategory::Real, 4>, std::int64_t);
CppTypeFor<TypeCategory::Real, 8> RTNAME(Scale8)(
    CppTypeFor<TypeCategory::Real, 8>, std::int64_t);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Scale10)(
    CppTypeFor<TypeCategory::Real, 10>, std::int64_t);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Scale16)(
    CppTypeFor<TypeCategory::Real, 16>, std::int64_t);
#endif

// SELECTED_REAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTNAME(SelectedRealKind)(
    const char *, int, void *, int, void *, int, void *, int);

// SPACING
CppTypeFor<TypeCategory::Real, 4> RTNAME(Spacing4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTNAME(Spacing8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Spacing10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Spacing16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_NUMERIC_H_
