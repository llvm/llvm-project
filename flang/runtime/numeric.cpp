//===-- runtime/numeric.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "numeric.h"
#include "flang/Common/long-double.h"
#include <climits>
#include <cmath>
#include <limits>

namespace Fortran::runtime {

// AINT
template <typename RESULT, typename ARG> inline RESULT Aint(ARG x) {
  return std::trunc(x);
}

// ANINT & NINT
template <typename RESULT, typename ARG> inline RESULT Anint(ARG x) {
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
    return 0; // 0 -> 0
  } else {
    int ignoredExp;
    return std::frexp(x, &ignoredExp);
  }
}

// MOD & MODULO (16.9.135, .136)
template <bool IS_MODULO, typename T> inline T IntMod(T x, T p) {
  auto mod{x - (x / p) * p};
  if (IS_MODULO && (x > 0) != (p > 0)) {
    mod += p;
  }
  return mod;
}
template <bool IS_MODULO, typename T> inline T RealMod(T x, T p) {
  if constexpr (IS_MODULO) {
    return x - std::floor(x / p) * p;
  } else {
    return x - std::trunc(x / p) * p;
  }
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

// SET_EXPONENT (16.9.171)
template <typename T> inline T SetExponent(T x, std::int64_t p) {
  if (std::isnan(x)) {
    return x; // NaN -> same NaN
  } else if (std::isinf(x)) {
    return std::numeric_limits<T>::quiet_NaN(); // +/-Inf -> NaN
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    int expo{std::ilogb(x)};
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
    return std::ldexp(
        static_cast<T>(1.0), std::ilogb(x) + 1 - PREC); // 2**(e-p)
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

extern "C" {

CppTypeFor<TypeCategory::Real, 4> RTNAME(Aint4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Aint4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Aint8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Aint8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Aint4_10)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
CppTypeFor<TypeCategory::Real, 10> RTNAME(Aint8_10)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Aint10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Aint10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 10> RTNAME(Aint10_10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Aint4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
CppTypeFor<TypeCategory::Real, 16> RTNAME(Aint8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Aint16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Aint16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 16> RTNAME(Aint16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Aint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(Anint4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Anint4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Anint8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Anint8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Anint4_10)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
CppTypeFor<TypeCategory::Real, 10> RTNAME(Anint8_10)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Anint10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Anint10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 10> RTNAME(Anint10_10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 10>>(x);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Anint4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
CppTypeFor<TypeCategory::Real, 16> RTNAME(Anint8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
CppTypeFor<TypeCategory::Real, 4> RTNAME(Anint16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 4>>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Anint16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 8>>(x);
}
CppTypeFor<TypeCategory::Real, 16> RTNAME(Anint16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Real, 16>>(x);
}
#endif

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
#ifdef __SIZEOF_INT128__
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
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LONG_DOUBLE == 80
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
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Ceiling10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Ceiling<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#else
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
#ifdef __SIZEOF_INT128__
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Exponent10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Exponent10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Exponent<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#elif LONG_DOUBLE == 128
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
#ifdef __SIZEOF_INT128__
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
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LONG_DOUBLE == 80
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
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Floor10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Floor<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#else
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
#ifdef __SIZEOF_INT128__
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Fraction10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Fraction(x);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Fraction16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Fraction(x);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p) {
  return IntMod<false>(x, p);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p) {
  return IntMod<false>(x, p);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p) {
  return IntMod<false>(x, p);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p) {
  return IntMod<false>(x, p);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p) {
  return IntMod<false>(x, p);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p) {
  return RealMod<false>(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p) {
  return RealMod<false>(x, p);
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModReal10)(
    CppTypeFor<TypeCategory::Real, 10> x,
    CppTypeFor<TypeCategory::Real, 10> p) {
  return RealMod<false>(x, p);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModReal16)(
    CppTypeFor<TypeCategory::Real, 16> x,
    CppTypeFor<TypeCategory::Real, 16> p) {
  return RealMod<false>(x, p);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(ModuloInteger1)(
    CppTypeFor<TypeCategory::Integer, 1> x,
    CppTypeFor<TypeCategory::Integer, 1> p) {
  return IntMod<true>(x, p);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ModuloInteger2)(
    CppTypeFor<TypeCategory::Integer, 2> x,
    CppTypeFor<TypeCategory::Integer, 2> p) {
  return IntMod<true>(x, p);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ModuloInteger4)(
    CppTypeFor<TypeCategory::Integer, 4> x,
    CppTypeFor<TypeCategory::Integer, 4> p) {
  return IntMod<true>(x, p);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ModuloInteger8)(
    CppTypeFor<TypeCategory::Integer, 8> x,
    CppTypeFor<TypeCategory::Integer, 8> p) {
  return IntMod<true>(x, p);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ModuloInteger16)(
    CppTypeFor<TypeCategory::Integer, 16> x,
    CppTypeFor<TypeCategory::Integer, 16> p) {
  return IntMod<true>(x, p);
}
#endif
CppTypeFor<TypeCategory::Real, 4> RTNAME(ModuloReal4)(
    CppTypeFor<TypeCategory::Real, 4> x, CppTypeFor<TypeCategory::Real, 4> p) {
  return RealMod<true>(x, p);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ModuloReal8)(
    CppTypeFor<TypeCategory::Real, 8> x, CppTypeFor<TypeCategory::Real, 8> p) {
  return RealMod<true>(x, p);
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(ModuloReal10)(
    CppTypeFor<TypeCategory::Real, 10> x,
    CppTypeFor<TypeCategory::Real, 10> p) {
  return RealMod<true>(x, p);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(ModuloReal16)(
    CppTypeFor<TypeCategory::Real, 16> x,
    CppTypeFor<TypeCategory::Real, 16> p) {
  return RealMod<true>(x, p);
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Nearest10)(
    CppTypeFor<TypeCategory::Real, 10> x, bool positive) {
  return Nearest<64>(x, positive);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Nearest16)(
    CppTypeFor<TypeCategory::Real, 16> x, bool positive) {
  return Nearest<113>(x, positive);
}
#endif

CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint4_1)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint4_2)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint4_4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint4_8)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint4_16)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint8_1)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint8_2)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint8_4)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint8_8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint8_16)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint10_1)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint10_2)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint10_4)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint10_8)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint10_16)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 16>>(x);
}
#endif
#else
CppTypeFor<TypeCategory::Integer, 1> RTNAME(Nint16_1)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 1>>(x);
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(Nint16_2)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 2>>(x);
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(Nint16_4)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 4>>(x);
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(Nint16_8)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 8>>(x);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(Nint16_16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Anint<CppTypeFor<TypeCategory::Integer, 16>>(x);
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(RRSpacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return RRSpacing<64>(x);
}
#elif LONG_DOUBLE == 128
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(SetExponent10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return SetExponent(x, p);
}
#elif LONG_DOUBLE == 128
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
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Scale10)(
    CppTypeFor<TypeCategory::Real, 10> x, std::int64_t p) {
  return Scale(x, p);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Scale16)(
    CppTypeFor<TypeCategory::Real, 16> x, std::int64_t p) {
  return Scale(x, p);
}
#endif

CppTypeFor<TypeCategory::Real, 4> RTNAME(Spacing4)(
    CppTypeFor<TypeCategory::Real, 4> x) {
  return Spacing<24>(x);
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Spacing8)(
    CppTypeFor<TypeCategory::Real, 8> x) {
  return Spacing<53>(x);
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Spacing10)(
    CppTypeFor<TypeCategory::Real, 10> x) {
  return Spacing<64>(x);
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Spacing16)(
    CppTypeFor<TypeCategory::Real, 16> x) {
  return Spacing<113>(x);
}
#endif
} // extern "C"
} // namespace Fortran::runtime
