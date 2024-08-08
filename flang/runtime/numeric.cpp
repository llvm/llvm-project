//===-- runtime/numeric.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/numeric.h"
#include "numeric-templates.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>

namespace Fortran::runtime {

template <typename RES>
inline RT_API_ATTRS RES GetIntArgValue(const char *source, int line,
    const void *arg, int kind, std::int64_t defaultValue, int resKind) {
  RES res;
  if (!arg) {
    res = static_cast<RES>(defaultValue);
  } else if (kind == 1) {
    res = static_cast<RES>(
        *static_cast<const CppTypeFor<TypeCategory::Integer, 1> *>(arg));
  } else if (kind == 2) {
    res = static_cast<RES>(
        *static_cast<const CppTypeFor<TypeCategory::Integer, 2> *>(arg));
  } else if (kind == 4) {
    res = static_cast<RES>(
        *static_cast<const CppTypeFor<TypeCategory::Integer, 4> *>(arg));
  } else if (kind == 8) {
    res = static_cast<RES>(
        *static_cast<const CppTypeFor<TypeCategory::Integer, 8> *>(arg));
#ifdef __SIZEOF_INT128__
  } else if (kind == 16) {
    if (resKind != 16) {
      Terminator{source, line}.Crash("Unexpected integer kind in runtime");
    }
    res = static_cast<RES>(
        *static_cast<const CppTypeFor<TypeCategory::Integer, 16> *>(arg));
#endif
  } else {
    Terminator{source, line}.Crash("Unexpected integer kind in runtime");
  }
  return res;
}

// NINT (16.9.141)
template <typename RESULT, typename ARG>
inline RT_API_ATTRS RESULT Nint(ARG x) {
  if (x >= 0) {
    return std::trunc(x + ARG{0.5});
  } else {
    return std::trunc(x - ARG{0.5});
  }
}

// CEILING & FLOOR (16.9.43, .79)
template <typename RESULT, typename ARG>
inline RT_API_ATTRS RESULT Ceiling(ARG x) {
  return std::ceil(x);
}
template <typename RESULT, typename ARG>
inline RT_API_ATTRS RESULT Floor(ARG x) {
  return std::floor(x);
}

// MOD & MODULO (16.9.135, .136)
template <bool IS_MODULO, typename T>
inline RT_API_ATTRS T IntMod(T x, T p, const char *sourceFile, int sourceLine) {
  if (p == 0) {
    Terminator{sourceFile, sourceLine}.Crash(
        IS_MODULO ? "MODULO with P==0" : "MOD with P==0");
  }
  auto mod{x - (x / p) * p};
  if (IS_MODULO && (x > 0) != (p > 0)) {
    mod += p;
  }
  return mod;
}

// SCALE (16.9.166)
template <typename T> inline RT_API_ATTRS T Scale(T x, std::int64_t p) {
  auto ip{static_cast<int>(p)};
  if (ip != p) {
    ip = p < 0 ? std::numeric_limits<int>::min()
               : std::numeric_limits<int>::max();
  }
  return std::ldexp(x, ip); // x*2**p
}

// SELECTED_INT_KIND (16.9.169)
template <typename T>
inline RT_API_ATTRS CppTypeFor<TypeCategory::Integer, 4> SelectedIntKind(T x) {
  if (x <= 2) {
    return 1;
  } else if (x <= 4) {
    return 2;
  } else if (x <= 9) {
    return 4;
  } else if (x <= 18) {
    return 8;
#ifdef __SIZEOF_INT128__
  } else if (x <= 38) {
    return 16;
#endif
  }
  return -1;
}

// SELECTED_LOGICAL_KIND (F'2023 16.9.182)
template <typename T>
inline RT_API_ATTRS CppTypeFor<TypeCategory::Integer, 4> SelectedLogicalKind(
    T x) {
  if (x <= 8) {
    return 1;
  } else if (x <= 16) {
    return 2;
  } else if (x <= 32) {
    return 4;
  } else if (x <= 64) {
    return 8;
  }
  return -1;
}

// SELECTED_REAL_KIND (16.9.170)
template <typename P, typename R, typename D>
inline RT_API_ATTRS CppTypeFor<TypeCategory::Integer, 4> SelectedRealKind(
    P p, R r, D d) {
  if (d != 2) {
    return -5;
  }

  int error{0};
  int kind{0};
  if (p <= 3) {
    kind = 2;
  } else if (p <= 6) {
    kind = 4;
  } else if (p <= 15) {
    kind = 8;
#if LDBL_MANT_DIG == 64
  } else if (p <= 18) {
    kind = 10;
  } else if (p <= 33) {
    kind = 16;
#elif LDBL_MANT_DIG == 113
  } else if (p <= 33) {
    kind = 16;
#endif
  } else {
    error -= 1;
  }

  if (r <= 4) {
    kind = kind < 2 ? 2 : kind;
  } else if (r <= 37) {
    kind = kind < 3 ? (p == 3 ? 4 : 3) : kind;
  } else if (r <= 307) {
    kind = kind < 8 ? 8 : kind;
#if LDBL_MANT_DIG == 64
  } else if (r <= 4931) {
    kind = kind < 10 ? 10 : kind;
#elif LDBL_MANT_DIG == 113
  } else if (r <= 4931) {
    kind = kind < 16 ? 16 : kind;
#endif
  } else {
    error -= 2;
  }

  return error ? error : kind;
}

// NEAREST (16.9.139)
template <int PREC, typename T>
inline RT_API_ATTRS T Nearest(T x, bool positive) {
  if (positive) {
    return std::nextafter(x, std::numeric_limits<T>::infinity());
  } else {
    return std::nextafter(x, -std::numeric_limits<T>::infinity());
  }
}

// Exponentiation operator for (Real ** Integer) cases (10.1.5.2.1).
template <typename BTy, typename ETy>
RT_API_ATTRS BTy FPowI(BTy base, ETy exp) {
  if (exp == ETy{0})
    return BTy{1};
  bool isNegativePower{exp < ETy{0}};
  bool isMinPower{exp == std::numeric_limits<ETy>::min()};
  if (isMinPower) {
    exp = std::numeric_limits<ETy>::max();
  } else if (isNegativePower) {
    exp = -exp;
  }
  BTy result{1};
  BTy origBase{base};
  while (true) {
    if (exp & ETy{1}) {
      result *= base;
    }
    exp >>= 1;
    if (exp == ETy{0}) {
      break;
    }
    base *= base;
  }
  if (isMinPower) {
    result *= origBase;
  }
  if (isNegativePower) {
    result = BTy{1} / result;
  }
  return result;
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(Ceiling4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Ceiling4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Ceiling4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Ceiling4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Ceiling4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Ceiling8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Ceiling8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Ceiling8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Ceiling8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Ceiling8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Ceiling10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Ceiling10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Ceiling10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Ceiling10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Ceiling10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Ceiling16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Ceiling16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Ceiling16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Ceiling16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Ceiling16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(ErfcScaled4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return ErfcScaled(x);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(ErfcScaled8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return ErfcScaled(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(ErfcScaled10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return ErfcScaled(x);
}
#endif
#if LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(ErfcScaled16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return ErfcScaled(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 4> RTDEF(Exponent4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Exponent4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Exponent8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Exponent8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Exponent10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Exponent10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTDEF(Floor4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Floor4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Floor4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Floor4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Floor4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Floor8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Floor8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Floor8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Floor8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Floor8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Floor10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Floor10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Floor10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Floor10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Floor10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Floor16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Floor16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Floor16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Floor16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Floor16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(Fraction4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Fraction(x);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Fraction8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Fraction(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(Fraction10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Fraction(x);
}
#endif

bool RTDEF(IsFinite4)(CppTypeFor<TypeCategory::Real, 4> x) {
  return std::isfinite(x);
}
bool RTDEF(IsFinite8)(CppTypeFor<TypeCategory::Real, 8> x) {
  return std::isfinite(x);
}
#if LDBL_MANT_DIG == 64
bool RTDEF(IsFinite10)(CppTypeFor<TypeCategory::Real, 10> x) {
  return std::isfinite(x);
}
#elif LDBL_MANT_DIG == 113
bool RTDEF(IsFinite16)(CppTypeFor<TypeCategory::Real, 16> x) {
  return std::isfinite(x);
}
#endif

bool RTDEF(IsNaN4)(CppTypeFor<TypeCategory::Real, 4> x) {
  return std::isnan(x);
}
bool RTDEF(IsNaN8)(CppTypeFor<TypeCategory::Real, 8> x) {
  return std::isnan(x);
}
#if LDBL_MANT_DIG == 64
bool RTDEF(IsNaN10)(CppTypeFor<TypeCategory::Real, 10> x) {
  return std::isnan(x);
}
#elif LDBL_MANT_DIG == 113
bool RTDEF(IsNaN16)(CppTypeFor<TypeCategory::Real, 16> x) {
  return std::isnan(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTDEF(ModInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(ModInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(ModInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(ModInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(ModInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTDEF(ModReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(ModReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(ModReal10)(
    CppTypeFor<TypeCategory::Real, 10> x, CppTypeFor<TypeCategory::Real, 10> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTDEF(ModuloInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(ModuloInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(ModuloInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(ModuloInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(ModuloInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTDEF(ModuloReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(ModuloReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(ModuloReal10)(
    CppTypeFor<TypeCategory::Real, 10> x, CppTypeFor<TypeCategory::Real, 10> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(Nearest4)(
    CppTypeFor<TypeCategory::Real, 4> x, bool positive) {
  return Nearest<24>(x, positive);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Nearest8)(
    CppTypeFor<TypeCategory::Real, 8> x, bool positive) {
  return Nearest<53>(x, positive);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(Nearest10)(
    CppTypeFor<TypeCategory::Real, 10> x, bool positive) {
  return Nearest<64>(x, positive);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTDEF(Nint4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Nint4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Nint4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Nint4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Nint4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Nint8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Nint8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Nint8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Nint8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Nint8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Nint10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Nint10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Nint10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Nint10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Nint10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTDEF(Nint16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(Nint16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Nint16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Nint16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTDEF(Nint16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(RRSpacing4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return RRSpacing<24>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(RRSpacing8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return RRSpacing<53>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(RRSpacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return RRSpacing<64>(x);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(SetExponent4)(
    CppTypeFor<TypeCategory::Real, 4> x, std::int64_t p) {
  return SetExponent(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(SetExponent8)(
    CppTypeFor<TypeCategory::Real, 8> x, std::int64_t p) {
  return SetExponent(x, p);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(SetExponent10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return SetExponent(x, p);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(Scale4)(
    CppTypeFor<TypeCategory::Real, 4> x, std::int64_t p) {
  return Scale(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Scale8)(
    CppTypeFor<TypeCategory::Real, 8> x, std::int64_t p) {
  return Scale(x, p);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(Scale10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return Scale(x, p);
}
#endif

// SELECTED_CHAR_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedCharKind)(
    const char *source, int line, const char *x, std::size_t length) {
  static const char *keywords[]{
      "ASCII", "DEFAULT", "UCS-2", "ISO_10646", "UCS-4", nullptr};
  switch (IdentifyValue(x, length, keywords)) {
  case 0: // ASCII
  case 1: // DEFAULT
    return 1;
  case 2: // UCS-2
    return 2;
  case 3: // ISO_10646
  case 4: // UCS-4
    return 4;
  default:
    return -1;
  }
}
// SELECTED_INT_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedIntKind)(
    const char *source, int line, void *x, int xKind) {
#ifdef __SIZEOF_INT128__
  CppTypeFor<TypeCategory::Integer, 16> r =
      GetIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 16);
#else
  std::int64_t r = GetIntArgValue<std::int64_t>(
      source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 8);
#endif
  return SelectedIntKind(r);
}

// SELECTED_LOGICAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedLogicalKind)(
    const char *source, int line, void *x, int xKind) {
#ifdef __SIZEOF_INT128__
  CppTypeFor<TypeCategory::Integer, 16> r =
      GetIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 16);
#else
  std::int64_t r = GetIntArgValue<std::int64_t>(
      source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 8);
#endif
  return SelectedLogicalKind(r);
}

// SELECTED_REAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedRealKind)(const char *source,
    int line, void *precision, int pKind, void *range, int rKind, void *radix,
    int dKind) {
#ifdef __SIZEOF_INT128__
  CppTypeFor<TypeCategory::Integer, 16> p =
      GetIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, precision, pKind, /*defaultValue*/ 0, /*resKind*/ 16);
  CppTypeFor<TypeCategory::Integer, 16> r =
      GetIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, range, rKind, /*defaultValue*/ 0, /*resKind*/ 16);
  CppTypeFor<TypeCategory::Integer, 16> d =
      GetIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, radix, dKind, /*defaultValue*/ 2, /*resKind*/ 16);
#else
  std::int64_t p = GetIntArgValue<std::int64_t>(
      source, line, precision, pKind, /*defaultValue*/ 0, /*resKind*/ 8);
  std::int64_t r = GetIntArgValue<std::int64_t>(
      source, line, range, rKind, /*defaultValue*/ 0, /*resKind*/ 8);
  std::int64_t d = GetIntArgValue<std::int64_t>(
      source, line, radix, dKind, /*defaultValue*/ 2, /*resKind*/ 8);
#endif
  return SelectedRealKind(p, r, d);
}

CppTypeFor<TypeCategory::Real, 4> RTDEF(Spacing4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Spacing<24>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Spacing8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Spacing<53>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(Spacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Spacing<64>(x);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(FPow4i)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(FPow8i)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(FPow10i)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(FPow16i)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTDEF(FPow4k)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(FPow8k)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(FPow10k)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(FPow16k)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
