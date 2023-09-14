//===-- runtime/numeric.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/numeric.h"
#include "terminator.h"
#include "flang/Runtime/float128.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>

namespace Fortran::runtime {

template <typename RES>
inline RES getIntArgValue(const char *source, int line, void *arg, int kind,
    std::int64_t defaultValue, int resKind) {
  RES res;
  if (!arg) {
    res = static_cast<RES>(defaultValue);
  } else if (kind == 1) {
    res = static_cast<RES>(
        *static_cast<CppTypeFor<TypeCategory::Integer, 1> *>(arg));
  } else if (kind == 2) {
    res = static_cast<RES>(
        *static_cast<CppTypeFor<TypeCategory::Integer, 2> *>(arg));
  } else if (kind == 4) {
    res = static_cast<RES>(
        *static_cast<CppTypeFor<TypeCategory::Integer, 4> *>(arg));
  } else if (kind == 8) {
    res = static_cast<RES>(
        *static_cast<CppTypeFor<TypeCategory::Integer, 8> *>(arg));
#ifdef __SIZEOF_INT128__
  } else if (kind == 16) {
    if (resKind != 16) {
      Terminator{source, line}.Crash("Unexpected integer kind in runtime");
    }
    res = static_cast<RES>(
        *static_cast<CppTypeFor<TypeCategory::Integer, 16> *>(arg));
#endif
  } else {
    Terminator{source, line}.Crash("Unexpected integer kind in runtime");
  }
  return res;
}

// NINT (16.9.141)
template <typename RESULT, typename ARG> inline RESULT Nint(ARG x) {
  if (x >= 0) {
    return std::trunc(x + ARG{0.5});
  } else {
    return std::trunc(x - ARG{0.5});
  }
}

// CEILING & FLOOR (16.9.43, .79)
template <typename RESULT, typename ARG> inline RESULT Ceiling(ARG x) {
  return std::ceil(x);
}
template <typename RESULT, typename ARG> inline RESULT Floor(ARG x) {
  return std::floor(x);
}

// EXPONENT (16.9.75)
template <typename RESULT, typename ARG> inline RESULT Exponent(ARG x) {
  if (std::isinf(x) || std::isnan(x)) {
    return std::numeric_limits<RESULT>::max(); // +/-Inf, NaN -> HUGE(0)
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    return std::ilogb(x) + 1;
  }
}

// FRACTION (16.9.80)
template <typename T> inline T Fraction(T x) {
  if (std::isnan(x)) {
    return x; // NaN -> same NaN
  } else if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN(); // +/-Inf -> NaN
  } else if (x == 0) {
    return x; // 0 -> same 0
  } else {
    int ignoredExp;
    return std::frexp(x, &ignoredExp);
  }
}

// MOD & MODULO (16.9.135, .136)
template <bool IS_MODULO, typename T>
inline T IntMod(T x, T p, const char *sourceFile, int sourceLine) {
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
template <bool IS_MODULO, typename T>
inline T RealMod(T a, T p, const char *sourceFile, int sourceLine) {
  if (p == 0) {
    Terminator{sourceFile, sourceLine}.Crash(
        IS_MODULO ? "MODULO with P==0" : "MOD with P==0");
  }
  T quotient{a / p};
  if (std::isinf(quotient) && std::isfinite(a) && std::isfinite(p)) {
    // a/p overflowed -- so it must be an integer, and the result
    // must be a zero of the same sign as one of the operands.
    return std::copysign(T{}, IS_MODULO ? p : a);
  }
  T toInt{IS_MODULO ? std::floor(quotient) : std::trunc(quotient)};
  return a - toInt * p;
}

// RRSPACING (16.9.164)
template <int PREC, typename T> inline T RRSpacing(T x) {
  if (std::isnan(x)) {
    return x; // NaN -> same NaN
  } else if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN(); // +/-Inf -> NaN
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    return std::ldexp(std::abs(x), PREC - (std::ilogb(x) + 1));
  }
}

// SCALE (16.9.166)
template <typename T> inline T Scale(T x, std::int64_t p) {
  auto ip{static_cast<int>(p)};
  if (ip != p) {
    ip = p < 0 ? std::numeric_limits<int>::min()
               : std::numeric_limits<int>::max();
  }
  return std::ldexp(x, p); // x*2**p
}

// SELECTED_INT_KIND (16.9.169)
template <typename T>
inline CppTypeFor<TypeCategory::Integer, 4> SelectedIntKind(T x) {
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

// SELECTED_REAL_KIND (16.9.170)
template <typename P, typename R, typename D>
inline CppTypeFor<TypeCategory::Integer, 4> SelectedRealKind(P p, R r, D d) {
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

// SET_EXPONENT (16.9.171)
template <typename T> inline T SetExponent(T x, std::int64_t p) {
  if (std::isnan(x)) {
    return x; // NaN -> same NaN
  } else if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN(); // +/-Inf -> NaN
  } else if (x == 0) {
    return x; // return negative zero if x is negative zero
  } else {
    int expo{std::ilogb(x) + 1};
    auto ip{static_cast<int>(p - expo)};
    if (ip != p - expo) {
      ip = p < 0 ? std::numeric_limits<int>::min()
                 : std::numeric_limits<int>::max();
    }
    return std::ldexp(x, ip); // x*2**(p-e)
  }
}

// SPACING (16.9.180)
template <int PREC, typename T> inline T Spacing(T x) {
  if (std::isnan(x)) {
    return x; // NaN -> same NaN
  } else if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN(); // +/-Inf -> NaN
  } else if (x == 0) {
    // The standard-mandated behavior seems broken, since TINY() can't be
    // subnormal.
    return std::numeric_limits<T>::min(); // 0 -> TINY(x)
  } else {
    T result{
        std::ldexp(static_cast<T>(1.0), std::ilogb(x) + 1 - PREC)}; // 2**(e-p)
    return result == 0 ? /*TINY(x)*/ std::numeric_limits<T>::min() : result;
  }
}

// NEAREST (16.9.139)
template <int PREC, typename T> inline T Nearest(T x, bool positive) {
  auto spacing{Spacing<PREC>(x)};
  if (x == 0) {
    auto least{std::numeric_limits<T>::denorm_min()};
    return positive ? least : -least;
  } else {
    return positive ? x + spacing : x - spacing;
  }
}

// Exponentiation operator for (Real ** Integer) cases (10.1.5.2.1).
template <typename BTy, typename ETy> BTy FPowI(BTy base, ETy exp) {
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

CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Ceiling16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Ceiling16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Ceiling16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Ceiling16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Floor16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Floor16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Floor16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Floor16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(Fraction4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Fraction(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Fraction8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Fraction(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Fraction10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Fraction(x);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(Fraction16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Fraction(x);
}
#endif

bool RTNAME(IsFinite4)(CppTypeFor<TypeCategory::Real, 4> x) {
  return std::isfinite(x);
}
bool RTNAME(IsFinite8)(CppTypeFor<TypeCategory::Real, 8> x) {
  return std::isfinite(x);
}
#if LDBL_MANT_DIG == 64
bool RTNAME(IsFinite10)(CppTypeFor<TypeCategory::Real, 10> x) {
  return std::isfinite(x);
}
#elif LDBL_MANT_DIG == 113
bool RTNAME(IsFinite16)(CppTypeFor<TypeCategory::Real, 16> x) {
  return std::isfinite(x);
}
#endif

bool RTNAME(IsNaN4)(CppTypeFor<TypeCategory::Real, 4> x) {
  return std::isnan(x);
}
bool RTNAME(IsNaN8)(CppTypeFor<TypeCategory::Real, 8> x) {
  return std::isnan(x);
}
#if LDBL_MANT_DIG == 64
bool RTNAME(IsNaN10)(CppTypeFor<TypeCategory::Real, 10> x) {
  return std::isnan(x);
}
#elif LDBL_MANT_DIG == 113
bool RTNAME(IsNaN16)(CppTypeFor<TypeCategory::Real, 16> x) {
  return std::isnan(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<false>(x, p, sourceFile, sourceLine);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModReal10)(
    CppTypeFor<TypeCategory::Real, 10> x, CppTypeFor<TypeCategory::Real, 10> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModReal16)(
    CppTypeFor<TypeCategory::Real, 16> x, CppTypeFor<TypeCategory::Real, 16> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<false>(x, p, sourceFile, sourceLine);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModuloInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModuloInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModuloInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModuloInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModuloInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p, const char *sourceFile,
    int sourceLine) {
  return IntMod<true>(x, p, sourceFile, sourceLine);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModuloReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModuloReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModuloReal10)(
    CppTypeFor<TypeCategory::Real, 10> x, CppTypeFor<TypeCategory::Real, 10> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModuloReal16)(
    CppTypeFor<TypeCategory::Real, 16> x, CppTypeFor<TypeCategory::Real, 16> p,
    const char *sourceFile, int sourceLine) {
  return RealMod<true>(x, p, sourceFile, sourceLine);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(Nearest4)(
    CppTypeFor<TypeCategory::Real, 4> x, bool positive) {
  return Nearest<24>(x, positive);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Nearest8)(
    CppTypeFor<TypeCategory::Real, 8> x, bool positive) {
  return Nearest<53>(x, positive);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Nearest10)(
    CppTypeFor<TypeCategory::Real, 10> x, bool positive) {
  return Nearest<64>(x, positive);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(Nearest16)(
    CppTypeFor<TypeCategory::Real, 16> x, bool positive) {
  return Nearest<113>(x, positive);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Nint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(RRSpacing4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return RRSpacing<24>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(RRSpacing8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return RRSpacing<53>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(RRSpacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return RRSpacing<64>(x);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(RRSpacing16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return RRSpacing<113>(x);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(SetExponent4)(
    CppTypeFor<TypeCategory::Real, 4> x, std::int64_t p) {
  return SetExponent(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(SetExponent8)(
    CppTypeFor<TypeCategory::Real, 8> x, std::int64_t p) {
  return SetExponent(x, p);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(SetExponent10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return SetExponent(x, p);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(SetExponent16)(
    CppTypeFor<TypeCategory::Real, 16> x, std::int64_t p) {
  return SetExponent(x, p);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(Scale4)(
    CppTypeFor<TypeCategory::Real, 4> x, std::int64_t p) {
  return Scale(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Scale8)(
    CppTypeFor<TypeCategory::Real, 8> x, std::int64_t p) {
  return Scale(x, p);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Scale10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return Scale(x, p);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(Scale16)(
    CppTypeFor<TypeCategory::Real, 16> x, std::int64_t p) {
  return Scale(x, p);
}
#endif

// SELECTED_INT_KIND
CppTypeFor<TypeCategory::Integer, 4> RTNAME(SelectedIntKind)(
    const char *source, int line, void *x, int xKind) {
#ifdef __SIZEOF_INT128__
  CppTypeFor<TypeCategory::Integer, 16> r =
      getIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 16);
#else
  std::int64_t r = getIntArgValue<std::int64_t>(
      source, line, x, xKind, /*defaultValue*/ 0, /*resKind*/ 8);
#endif
  return SelectedIntKind(r);
}

// SELECTED_REAL_KIND
CppTypeFor<TypeCategory::Integer, 4> RTNAME(SelectedRealKind)(
    const char *source, int line, void *precision, int pKind, void *range,
    int rKind, void *radix, int dKind) {
#ifdef __SIZEOF_INT128__
  CppTypeFor<TypeCategory::Integer, 16> p =
      getIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, precision, pKind, /*defaultValue*/ 0, /*resKind*/ 16);
  CppTypeFor<TypeCategory::Integer, 16> r =
      getIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, range, rKind, /*defaultValue*/ 0, /*resKind*/ 16);
  CppTypeFor<TypeCategory::Integer, 16> d =
      getIntArgValue<CppTypeFor<TypeCategory::Integer, 16>>(
          source, line, radix, dKind, /*defaultValue*/ 2, /*resKind*/ 16);
#else
  std::int64_t p = getIntArgValue<std::int64_t>(
      source, line, precision, pKind, /*defaultValue*/ 0, /*resKind*/ 8);
  std::int64_t r = getIntArgValue<std::int64_t>(
      source, line, range, rKind, /*defaultValue*/ 0, /*resKind*/ 8);
  std::int64_t d = getIntArgValue<std::int64_t>(
      source, line, radix, dKind, /*defaultValue*/ 2, /*resKind*/ 8);
#endif
  return SelectedRealKind(p, r, d);
}

CppTypeFor<TypeCategory::Real, 4> RTNAME(Spacing4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Spacing<24>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Spacing8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Spacing<53>(x);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(Spacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Spacing<64>(x);
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(Spacing16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Spacing<113>(x);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(FPow4i)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(FPow8i)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(FPow10i)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(FPow16i)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 4> e) {
  return FPowI(b, e);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(FPow4k)(
    CppTypeFor<TypeCategory::Real, 4> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(FPow8k)(
    CppTypeFor<TypeCategory::Real, 8> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(FPow10k)(
    CppTypeFor<TypeCategory::Real, 10> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTNAME(FPow16k)(
    CppTypeFor<TypeCategory::Real, 16> b,
    CppTypeFor<TypeCategory::Integer, 8> e) {
  return FPowI(b, e);
}
#endif
} // extern "C"
} // namespace Fortran::runtime
