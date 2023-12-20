//===-- runtime/numeric.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/numeric.h"
#include "terminator.h"
#include "flang/Common/float128.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>

namespace Fortran::runtime {

template <typename RES>
inline RT_API_ATTRS RES getIntArgValue(const char *source, int line, void *arg,
    int kind, std::int64_t defaultValue, int resKind) {
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

// EXPONENT (16.9.75)
template <typename RESULT, typename ARG>
inline RT_API_ATTRS RESULT Exponent(ARG x) {
  if (std::isinf(x) || std::isnan(x)) {
    return std::numeric_limits<RESULT>::max(); // +/-Inf, NaN -> HUGE(0)
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    return std::ilogb(x) + 1;
  }
}

// Suppress the warnings about calling __host__-only std::frexp,
// defined in C++ STD header files, from __device__ code.
RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

// FRACTION (16.9.80)
template <typename T> inline RT_API_ATTRS T Fraction(T x) {
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

RT_DIAG_POP

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
template <bool IS_MODULO, typename T>
inline RT_API_ATTRS T RealMod(
    T a, T p, const char *sourceFile, int sourceLine) {
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
template <int PREC, typename T> inline RT_API_ATTRS T RRSpacing(T x) {
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
template <typename T> inline RT_API_ATTRS T Scale(T x, std::int64_t p) {
  auto ip{static_cast<int>(p)};
  if (ip != p) {
    ip = p < 0 ? std::numeric_limits<int>::min()
               : std::numeric_limits<int>::max();
  }
  return std::ldexp(x, p); // x*2**p
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

// SET_EXPONENT (16.9.171)
template <typename T> inline RT_API_ATTRS T SetExponent(T x, std::int64_t p) {
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
template <int PREC, typename T> inline RT_API_ATTRS T Spacing(T x) {
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
template <int PREC, typename T>
inline RT_API_ATTRS T Nearest(T x, bool positive) {
  auto spacing{Spacing<PREC>(x)};
  if (x == 0) {
    auto least{std::numeric_limits<T>::denorm_min()};
    return positive ? least : -least;
  } else {
    return positive ? x + spacing : x - spacing;
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Integer, 4> RTDEF(Exponent16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(Exponent16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(Fraction16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(ModReal16)(
    CppTypeFor<TypeCategory::Real, 16> x, CppTypeFor<TypeCategory::Real, 16> p,
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(ModuloReal16)(
    CppTypeFor<TypeCategory::Real, 16> x, CppTypeFor<TypeCategory::Real, 16> p,
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(Nearest16)(
    CppTypeFor<TypeCategory::Real, 16> x, bool positive) {
  return Nearest<113>(x, positive);
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(RRSpacing16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return RRSpacing<113>(x);
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(SetExponent16)(
    CppTypeFor<TypeCategory::Real, 16> x, std::int64_t p) {
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(Scale16)(
    CppTypeFor<TypeCategory::Real, 16> x, std::int64_t p) {
  return Scale(x, p);
}
#endif

// SELECTED_INT_KIND
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedIntKind)(
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
CppTypeFor<TypeCategory::Integer, 4> RTDEF(SelectedRealKind)(const char *source,
    int line, void *precision, int pKind, void *range, int rKind, void *radix,
    int dKind) {
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
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTDEF(Spacing16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Spacing<113>(x);
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
} // extern "C"
} // namespace Fortran::runtime
