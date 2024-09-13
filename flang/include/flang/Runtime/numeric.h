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

#include "flang/Common/float128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/entry-names.h"

namespace Fortran::runtime {
extern "C" {

// CEILING
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Ceiling4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Ceiling4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Ceiling4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Ceiling4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Ceiling4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Ceiling8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Ceiling8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Ceiling8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Ceiling8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Ceiling8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Ceiling10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Ceiling10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Ceiling10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Ceiling10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Ceiling10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Ceiling16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Ceiling16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Ceiling16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Ceiling16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Ceiling16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// ERFC_SCALED
CppTypeFor<TypeCategory::Real, 4> RTDECL(ErfcScaled4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTDECL(ErfcScaled8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(ErfcScaled10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(ErfcScaled16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// EXPONENT is defined to return default INTEGER; support INTEGER(4 & 8)
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Exponent4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Exponent4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Exponent8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Exponent8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Exponent10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Exponent10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Exponent16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Exponent16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// FLOOR
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Floor4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Floor4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Floor4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Floor4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Floor4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Floor8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Floor8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Floor8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Floor8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Floor8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Floor10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Floor10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Floor10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Floor10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Floor10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Floor16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Floor16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Floor16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Floor16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Floor16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// FRACTION
CppTypeFor<TypeCategory::Real, 4> RTDECL(Fraction4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTDECL(Fraction8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(Fraction10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(Fraction16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// ISNAN / IEEE_IS_NAN
bool RTDECL(IsNaN4)(CppTypeFor<TypeCategory::Real, 4>);
bool RTDECL(IsNaN8)(CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
bool RTDECL(IsNaN10)(CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
bool RTDECL(IsNaN16)(CppTypeFor<TypeCategory::Real, 16>);
#endif

// MOD & MODULO
CppTypeFor<TypeCategory::Integer, 1> RTDECL(ModInteger1)(
    CppTypeFor<TypeCategory::Integer, 1>, CppTypeFor<TypeCategory::Integer, 1>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(ModInteger2)(
    CppTypeFor<TypeCategory::Integer, 2>, CppTypeFor<TypeCategory::Integer, 2>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(ModInteger4)(
    CppTypeFor<TypeCategory::Integer, 4>, CppTypeFor<TypeCategory::Integer, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(ModInteger8)(
    CppTypeFor<TypeCategory::Integer, 8>, CppTypeFor<TypeCategory::Integer, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(ModInteger16)(
    CppTypeFor<TypeCategory::Integer, 16>,
    CppTypeFor<TypeCategory::Integer, 16>, const char *sourceFile = nullptr,
    int sourceLine = 0);
#endif
CppTypeFor<TypeCategory::Real, 4> RTDECL(ModReal4)(
    CppTypeFor<TypeCategory::Real, 4>, CppTypeFor<TypeCategory::Real, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Real, 8> RTDECL(ModReal8)(
    CppTypeFor<TypeCategory::Real, 8>, CppTypeFor<TypeCategory::Real, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(ModReal10)(
    CppTypeFor<TypeCategory::Real, 10>, CppTypeFor<TypeCategory::Real, 10>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(ModReal16)(
    CppTypeFor<TypeCategory::Real, 16>, CppTypeFor<TypeCategory::Real, 16>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif

CppTypeFor<TypeCategory::Integer, 1> RTDECL(ModuloInteger1)(
    CppTypeFor<TypeCategory::Integer, 1>, CppTypeFor<TypeCategory::Integer, 1>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(ModuloInteger2)(
    CppTypeFor<TypeCategory::Integer, 2>, CppTypeFor<TypeCategory::Integer, 2>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(ModuloInteger4)(
    CppTypeFor<TypeCategory::Integer, 4>, CppTypeFor<TypeCategory::Integer, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(ModuloInteger8)(
    CppTypeFor<TypeCategory::Integer, 8>, CppTypeFor<TypeCategory::Integer, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(ModuloInteger16)(
    CppTypeFor<TypeCategory::Integer, 16>,
    CppTypeFor<TypeCategory::Integer, 16>, const char *sourceFile = nullptr,
    int sourceLine = 0);
#endif
CppTypeFor<TypeCategory::Real, 4> RTDECL(ModuloReal4)(
    CppTypeFor<TypeCategory::Real, 4>, CppTypeFor<TypeCategory::Real, 4>,
    const char *sourceFile = nullptr, int sourceLine = 0);
CppTypeFor<TypeCategory::Real, 8> RTDECL(ModuloReal8)(
    CppTypeFor<TypeCategory::Real, 8>, CppTypeFor<TypeCategory::Real, 8>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(ModuloReal10)(
    CppTypeFor<TypeCategory::Real, 10>, CppTypeFor<TypeCategory::Real, 10>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(ModuloReal16)(
    CppTypeFor<TypeCategory::Real, 16>, CppTypeFor<TypeCategory::Real, 16>,
    const char *sourceFile = nullptr, int sourceLine = 0);
#endif

// NINT
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Nint4_1)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Nint4_2)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Nint4_4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Nint4_8)(
    CppTypeFor<TypeCategory::Real, 4>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Nint4_16)(
    CppTypeFor<TypeCategory::Real, 4>);
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Nint8_1)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Nint8_2)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Nint8_4)(
    CppTypeFor<TypeCategory::Real, 8>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Nint8_8)(
    CppTypeFor<TypeCategory::Real, 8>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Nint8_16)(
    CppTypeFor<TypeCategory::Real, 8>);
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Nint10_1)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Nint10_2)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Nint10_4)(
    CppTypeFor<TypeCategory::Real, 10>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Nint10_8)(
    CppTypeFor<TypeCategory::Real, 10>);
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Nint10_16)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Integer, 1> RTDECL(Nint16_1)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 2> RTDECL(Nint16_2)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(Nint16_4)(
    CppTypeFor<TypeCategory::Real, 16>);
CppTypeFor<TypeCategory::Integer, 8> RTDECL(Nint16_8)(
    CppTypeFor<TypeCategory::Real, 16>);
#if defined __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDECL(Nint16_16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif
#endif

// NEAREST
// The second argument to NEAREST is the result of a comparison
// to zero (i.e., S > 0)
CppTypeFor<TypeCategory::Real, 4> RTDECL(Nearest4)(
    CppTypeFor<TypeCategory::Real, 4>, bool positive);
CppTypeFor<TypeCategory::Real, 8> RTDECL(Nearest8)(
    CppTypeFor<TypeCategory::Real, 8>, bool positive);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(Nearest10)(
    CppTypeFor<TypeCategory::Real, 10>, bool positive);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(Nearest16)(
    CppTypeFor<TypeCategory::Real, 16>, bool positive);
#endif

// RRSPACING
CppTypeFor<TypeCategory::Real, 4> RTDECL(RRSpacing4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTDECL(RRSpacing8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(RRSpacing10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(RRSpacing16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

// SET_EXPONENT's I= argument can be any INTEGER kind; upcast it to 64-bit
CppTypeFor<TypeCategory::Real, 4> RTDECL(SetExponent4)(
    CppTypeFor<TypeCategory::Real, 4>, std::int64_t);
CppTypeFor<TypeCategory::Real, 8> RTDECL(SetExponent8)(
    CppTypeFor<TypeCategory::Real, 8>, std::int64_t);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(SetExponent10)(
    CppTypeFor<TypeCategory::Real, 10>, std::int64_t);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(SetExponent16)(
    CppTypeFor<TypeCategory::Real, 16>, std::int64_t);
#endif

// SCALE
CppTypeFor<TypeCategory::Real, 4> RTDECL(Scale4)(
    CppTypeFor<TypeCategory::Real, 4>, std::int64_t);
CppTypeFor<TypeCategory::Real, 8> RTDECL(Scale8)(
    CppTypeFor<TypeCategory::Real, 8>, std::int64_t);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(Scale10)(
    CppTypeFor<TypeCategory::Real, 10>, std::int64_t);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(Scale16)(
    CppTypeFor<TypeCategory::Real, 16>, std::int64_t);
#endif

// SELECTED_CHAR_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedCharKind)(
    const char *, int, const char *, std::size_t);

// SELECTED_INT_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedIntKind)(
    const char *, int, void *, int);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedIntKindMasked)(
    const char *, int, void *, int, int);

// SELECTED_LOGICAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedLogicalKind)(
    const char *, int, void *, int);

// SELECTED_REAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedRealKind)(
    const char *, int, void *, int, void *, int, void *, int);
CppTypeFor<TypeCategory::Integer, 4> RTDECL(SelectedRealKindMasked)(
    const char *, int, void *, int, void *, int, void *, int, int);

// SPACING
// The variants Spacing2By4 and Spacing3By4 compute SPACING for REAL(2/3)
// but accept and return REAL(4) values, for use in environments where
// std::float16_t or std::bfloat16_t are unavailable.
#if HAS_FP16
CppTypeFor<TypeCategory::Real, 2> RTDECL(Spacing2)(
    CppTypeFor<TypeCategory::Real, 2>);
#endif
CppTypeFor<TypeCategory::Real, 4> RTDECL(Spacing2By4)(
    CppTypeFor<TypeCategory::Real, 4>);
#if HAS_BF16
CppTypeFor<TypeCategory::Real, 3> RTDECL(Spacing3)(
    CppTypeFor<TypeCategory::Real, 3>);
#endif
CppTypeFor<TypeCategory::Real, 4> RTDECL(Spacing3By4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 4> RTDECL(Spacing4)(
    CppTypeFor<TypeCategory::Real, 4>);
CppTypeFor<TypeCategory::Real, 8> RTDECL(Spacing8)(
    CppTypeFor<TypeCategory::Real, 8>);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(Spacing10)(
    CppTypeFor<TypeCategory::Real, 10>);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(Spacing16)(
    CppTypeFor<TypeCategory::Real, 16>);
#endif

CppTypeFor<TypeCategory::Real, 4> RTDECL(FPow4i)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 4> e);
CppTypeFor<TypeCategory::Real, 8> RTDECL(FPow8i)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 4> e);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(FPow10i)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 4> e);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(FPow16i)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 4> e);
#endif

CppTypeFor<TypeCategory::Real, 4> RTDECL(FPow4k)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 8> e);
CppTypeFor<TypeCategory::Real, 8> RTDECL(FPow8k)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 8> e);
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDECL(FPow10k)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 8> e);
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDECL(FPow16k)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 8> e);
#endif

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_NUMERIC_H_
