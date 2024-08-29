//===-- runtime/numeric-templates.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Generic class and function templates used for implementing
// various numeric intrinsics (EXPONENT, FRACTION, etc.).
//
// This header file also defines generic templates for "basic"
// math operations like abs, isnan, etc. The Float128Math
// library provides specializations for these templates
// for the data type corresponding to CppTypeFor<TypeCategory::Real, 16>
// on the target.

#ifndef FORTRAN_RUNTIME_NUMERIC_TEMPLATES_H_
#define FORTRAN_RUNTIME_NUMERIC_TEMPLATES_H_

#include "terminator.h"
#include "tools.h"
#include "flang/Common/api-attrs.h"
#include "flang/Common/float128.h"
#include <cstdint>
#include <limits>

namespace Fortran::runtime {

// MAX/MIN/LOWEST values for different data types.

// MaxOrMinIdentity returns MAX or LOWEST value of the given type.
template <TypeCategory CAT, int KIND, bool IS_MAXVAL, typename Enable = void>
struct MaxOrMinIdentity {
  using Type = CppTypeFor<CAT, KIND>;
  static constexpr RT_API_ATTRS Type Value() {
    return IS_MAXVAL ? std::numeric_limits<Type>::lowest()
                     : std::numeric_limits<Type>::max();
  }
};

// std::numeric_limits<> may not know int128_t
template <bool IS_MAXVAL>
struct MaxOrMinIdentity<TypeCategory::Integer, 16, IS_MAXVAL> {
  using Type = CppTypeFor<TypeCategory::Integer, 16>;
  static constexpr RT_API_ATTRS Type Value() {
    return IS_MAXVAL ? Type{1} << 127 : ~Type{0} >> 1;
  }
};

#if HAS_FLOAT128
// std::numeric_limits<> may not support __float128.
//
// Usage of GCC quadmath.h's FLT128_MAX is complicated by the fact that
// even GCC complains about 'Q' literal suffix under -Wpedantic.
// We just recreate FLT128_MAX ourselves.
//
// This specialization must engage only when
// CppTypeFor<TypeCategory::Real, 16> is __float128.
template <bool IS_MAXVAL>
struct MaxOrMinIdentity<TypeCategory::Real, 16, IS_MAXVAL,
    typename std::enable_if_t<
        std::is_same_v<CppTypeFor<TypeCategory::Real, 16>, __float128>>> {
  using Type = __float128;
  static RT_API_ATTRS Type Value() {
    // Create a buffer to store binary representation of __float128 constant.
    constexpr std::size_t alignment =
        std::max(alignof(Type), alignof(std::uint64_t));
    alignas(alignment) char data[sizeof(Type)];

    // First, verify that our interpretation of __float128 format is correct,
    // e.g. by checking at least one known constant.
    *reinterpret_cast<Type *>(data) = Type(1.0);
    if (*reinterpret_cast<std::uint64_t *>(data) != 0 ||
        *(reinterpret_cast<std::uint64_t *>(data) + 1) != 0x3FFF000000000000) {
      Terminator terminator{__FILE__, __LINE__};
      terminator.Crash("not yet implemented: no full support for __float128");
    }

    // Recreate FLT128_MAX.
    *reinterpret_cast<std::uint64_t *>(data) = 0xFFFFFFFFFFFFFFFF;
    *(reinterpret_cast<std::uint64_t *>(data) + 1) = 0x7FFEFFFFFFFFFFFF;
    Type max = *reinterpret_cast<Type *>(data);
    return IS_MAXVAL ? -max : max;
  }
};
#endif // HAS_FLOAT128

// Minimum finite representable value.
// For floating-point types, returns minimum positive normalized value.
template <typename T> struct MinValue {
  static RT_API_ATTRS T get() { return std::numeric_limits<T>::min(); }
};

#if HAS_FLOAT128
template <> struct MinValue<CppTypeFor<TypeCategory::Real, 16>> {
  using Type = CppTypeFor<TypeCategory::Real, 16>;
  static RT_API_ATTRS Type get() {
    // Create a buffer to store binary representation of __float128 constant.
    constexpr std::size_t alignment =
        std::max(alignof(Type), alignof(std::uint64_t));
    alignas(alignment) char data[sizeof(Type)];

    // First, verify that our interpretation of __float128 format is correct,
    // e.g. by checking at least one known constant.
    *reinterpret_cast<Type *>(data) = Type(1.0);
    if (*reinterpret_cast<std::uint64_t *>(data) != 0 ||
        *(reinterpret_cast<std::uint64_t *>(data) + 1) != 0x3FFF000000000000) {
      Terminator terminator{__FILE__, __LINE__};
      terminator.Crash("not yet implemented: no full support for __float128");
    }

    // Recreate FLT128_MIN.
    *reinterpret_cast<std::uint64_t *>(data) = 0;
    *(reinterpret_cast<std::uint64_t *>(data) + 1) = 0x1000000000000;
    return *reinterpret_cast<Type *>(data);
  }
};
#endif // HAS_FLOAT128

template <typename T> struct ABSTy {
  static constexpr RT_API_ATTRS T compute(T x) { return std::abs(x); }
};

// Suppress the warnings about calling __host__-only
// 'long double' std::frexp, from __device__ code.
RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

template <typename T> struct FREXPTy {
  static constexpr RT_API_ATTRS T compute(T x, int *e) {
    return std::frexp(x, e);
  }
};

RT_DIAG_POP

template <typename T> struct ILOGBTy {
  static constexpr RT_API_ATTRS int compute(T x) { return std::ilogb(x); }
};

template <typename T> struct ISINFTy {
  static constexpr RT_API_ATTRS bool compute(T x) { return std::isinf(x); }
};

template <typename T> struct ISNANTy {
  static constexpr RT_API_ATTRS bool compute(T x) { return std::isnan(x); }
};

template <typename T> struct LDEXPTy {
  template <typename ET> static constexpr RT_API_ATTRS T compute(T x, ET e) {
    return std::ldexp(x, e);
  }
};

template <typename T> struct MAXTy {
  static constexpr RT_API_ATTRS T compute() {
    return std::numeric_limits<T>::max();
  }
};

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
template <> struct MAXTy<CppTypeFor<TypeCategory::Real, 16>> {
  static CppTypeFor<TypeCategory::Real, 16> compute() {
    return MaxOrMinIdentity<TypeCategory::Real, 16, true>::Value();
  }
};
#endif

template <typename T> struct MINTy {
  static constexpr RT_API_ATTRS T compute() { return MinValue<T>::get(); }
};

template <typename T> struct QNANTy {
  static constexpr RT_API_ATTRS T compute() {
    return std::numeric_limits<T>::quiet_NaN();
  }
};

template <typename T> struct SQRTTy {
  static constexpr RT_API_ATTRS T compute(T x) { return std::sqrt(x); }
};

// EXPONENT (16.9.75)
template <typename RESULT, typename ARG>
inline RT_API_ATTRS RESULT Exponent(ARG x) {
  if (ISINFTy<ARG>::compute(x) || ISNANTy<ARG>::compute(x)) {
    return MAXTy<RESULT>::compute(); // +/-Inf, NaN -> HUGE(0)
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    return ILOGBTy<ARG>::compute(x) + 1;
  }
}

// FRACTION (16.9.80)
template <typename T> inline RT_API_ATTRS T Fraction(T x) {
  if (ISNANTy<T>::compute(x)) {
    return x; // NaN -> same NaN
  } else if (ISINFTy<T>::compute(x)) {
    return QNANTy<T>::compute(); // +/-Inf -> NaN
  } else if (x == 0) {
    return x; // 0 -> same 0
  } else {
    int ignoredExp;
    return FREXPTy<T>::compute(x, &ignoredExp);
  }
}

// SET_EXPONENT (16.9.171)
template <typename T> inline RT_API_ATTRS T SetExponent(T x, std::int64_t p) {
  if (ISNANTy<T>::compute(x)) {
    return x; // NaN -> same NaN
  } else if (ISINFTy<T>::compute(x)) {
    return QNANTy<T>::compute(); // +/-Inf -> NaN
  } else if (x == 0) {
    return x; // return negative zero if x is negative zero
  } else {
    int expo{ILOGBTy<T>::compute(x) + 1};
    auto ip{static_cast<int>(p - expo)};
    if (ip != p - expo) {
      ip = p < 0 ? std::numeric_limits<int>::min()
                 : std::numeric_limits<int>::max();
    }
    return LDEXPTy<T>::compute(x, ip); // x*2**(p-e)
  }
}

// MOD & MODULO (16.9.135, .136)
template <bool IS_MODULO, typename T>
inline RT_API_ATTRS T RealMod(
    T a, T p, const char *sourceFile, int sourceLine) {
  if (p == 0) {
    Terminator{sourceFile, sourceLine}.Crash(
        IS_MODULO ? "MODULO with P==0" : "MOD with P==0");
  }
  if (ISNANTy<T>::compute(a) || ISNANTy<T>::compute(p) ||
      ISINFTy<T>::compute(a)) {
    return QNANTy<T>::compute();
  } else if (IS_MODULO && ISINFTy<T>::compute(p)) {
    // Other compilers behave consistently for MOD(x, +/-INF)
    // and always return x. This is probably related to
    // implementation of std::fmod(). Stick to this behavior
    // for MOD, but return NaN for MODULO(x, +/-INF).
    return QNANTy<T>::compute();
  }
  T aAbs{ABSTy<T>::compute(a)};
  T pAbs{ABSTy<T>::compute(p)};
  if (aAbs <= static_cast<T>(std::numeric_limits<std::int64_t>::max()) &&
      pAbs <= static_cast<T>(std::numeric_limits<std::int64_t>::max())) {
    if (auto aInt{static_cast<std::int64_t>(a)}; a == aInt) {
      if (auto pInt{static_cast<std::int64_t>(p)}; p == pInt) {
        // Fast exact case for integer operands
        auto mod{aInt - (aInt / pInt) * pInt};
        if constexpr (IS_MODULO) {
          if (mod == 0) {
            // Return properly signed zero.
            return pInt > 0 ? T{0} : -T{0};
          }
          if ((aInt > 0) != (pInt > 0)) {
            mod += pInt;
          }
        } else {
          if (mod == 0) {
            // Return properly signed zero.
            return aInt > 0 ? T{0} : -T{0};
          }
        }
        return static_cast<T>(mod);
      }
    }
  }
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, long double>) {
    // std::fmod() semantics on signed operands seems to match
    // the requirements of MOD().  MODULO() needs adjustment.
    T result{std::fmod(a, p)};
    if constexpr (IS_MODULO) {
      if ((a < 0) != (p < 0)) {
        if (result == 0.) {
          result = -result;
        } else {
          result += p;
        }
      }
    }
    return result;
  } else {
    // The standard defines MOD(a,p)=a-AINT(a/p)*p and
    // MODULO(a,p)=a-FLOOR(a/p)*p, but those definitions lose
    // precision badly due to cancellation when ABS(a) is
    // much larger than ABS(p).
    // Insights:
    //  - MOD(a,p)=MOD(a-n*p,p) when a>0, p>0, integer n>0, and a>=n*p
    //  - when n is a power of two, n*p is exact
    //  - as a>=n*p, a-n*p does not round.
    // So repeatedly reduce a by all n*p in decreasing order of n;
    // what's left is the desired remainder.  This is basically
    // the same algorithm as arbitrary precision binary long division,
    // discarding the quotient.
    T tmp{aAbs};
    for (T adj{SetExponent(pAbs, Exponent<int>(aAbs))}; tmp >= pAbs; adj /= 2) {
      if (tmp >= adj) {
        tmp -= adj;
        if (tmp == 0) {
          break;
        }
      }
    }
    if (a < 0) {
      tmp = -tmp;
    }
    if constexpr (IS_MODULO) {
      if ((a < 0) != (p < 0)) {
        if (tmp == 0.) {
          tmp = -tmp;
        } else {
          tmp += p;
        }
      }
    }
    return tmp;
  }
}

// RRSPACING (16.9.164)
template <int PREC, typename T> inline RT_API_ATTRS T RRSpacing(T x) {
  if (ISNANTy<T>::compute(x)) {
    return x; // NaN -> same NaN
  } else if (ISINFTy<T>::compute(x)) {
    return QNANTy<T>::compute(); // +/-Inf -> NaN
  } else if (x == 0) {
    return 0; // 0 -> 0
  } else {
    return LDEXPTy<T>::compute(
        ABSTy<T>::compute(x), PREC - (ILOGBTy<T>::compute(x) + 1));
  }
}

// SPACING (16.9.180)
template <int PREC, typename T> inline RT_API_ATTRS T Spacing(T x) {
  if (ISNANTy<T>::compute(x)) {
    return x; // NaN -> same NaN
  } else if (ISINFTy<T>::compute(x)) {
    return QNANTy<T>::compute(); // +/-Inf -> NaN
  } else if (x == 0) {
    // The standard-mandated behavior seems broken, since TINY() can't be
    // subnormal.
    return MINTy<T>::compute(); // 0 -> TINY(x)
  } else {
    T result{LDEXPTy<T>::compute(
        static_cast<T>(1.0), ILOGBTy<T>::compute(x) + 1 - PREC)}; // 2**(e-p)
    return result == 0 ? /*TINY(x)*/ MINTy<T>::compute() : result;
  }
}

// ERFC_SCALED (16.9.71)
template <typename T> inline RT_API_ATTRS T ErfcScaled(T arg) {
  // Coefficients for approximation to erfc in the first interval.
  static const T a[5] = {3.16112374387056560e00, 1.13864154151050156e02,
      3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1};
  static const T b[4] = {2.36012909523441209e01, 2.44024637934444173e02,
      1.28261652607737228e03, 2.84423683343917062e03};

  // Coefficients for approximation to erfc in the second interval.
  static const T c[9] = {5.64188496988670089e-1, 8.88314979438837594e00,
      6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02,
      1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03,
      2.15311535474403846e-8};
  static const T d[8] = {1.57449261107098347e01, 1.17693950891312499e02,
      5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03,
      4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};

  // Coefficients for approximation to erfc in the third interval.
  static const T p[6] = {3.05326634961232344e-1, 3.60344899949804439e-1,
      1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4,
      1.63153871373020978e-2};
  static const T q[5] = {2.56852019228982242e00, 1.87295284992346047e00,
      5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3};

  constexpr T sqrtpi{1.7724538509078120380404576221783883301349L};
  constexpr T rsqrtpi{0.5641895835477562869480794515607725858440L};
  constexpr T epsilonby2{std::numeric_limits<T>::epsilon() * 0.5};
  constexpr T xneg{-26.628e0};
  constexpr T xhuge{6.71e7};
  constexpr T thresh{0.46875e0};
  constexpr T zero{0.0};
  constexpr T one{1.0};
  constexpr T four{4.0};
  constexpr T sixteen{16.0};
  constexpr T xmax{1.0 / (sqrtpi * std::numeric_limits<T>::min())};
  static_assert(xmax > xhuge, "xmax must be greater than xhuge");

  T ysq;
  T xnum;
  T xden;
  T del;
  T result;

  auto x{arg};
  auto y{std::fabs(x)};

  if (y <= thresh) {
    // evaluate erf for  |x| <= 0.46875
    ysq = zero;
    if (y > epsilonby2) {
      ysq = y * y;
    }
    xnum = a[4] * ysq;
    xden = ysq;
    for (int i{0}; i < 3; i++) {
      xnum = (xnum + a[i]) * ysq;
      xden = (xden + b[i]) * ysq;
    }
    result = x * (xnum + a[3]) / (xden + b[3]);
    result = one - result;
    result = std::exp(ysq) * result;
    return result;
  } else if (y <= four) {
    //  evaluate erfc for 0.46875 < |x| <= 4.0
    xnum = c[8] * y;
    xden = y;
    for (int i{0}; i < 7; ++i) {
      xnum = (xnum + c[i]) * y;
      xden = (xden + d[i]) * y;
    }
    result = (xnum + c[7]) / (xden + d[7]);
  } else {
    //  evaluate erfc for |x| > 4.0
    result = zero;
    if (y >= xhuge) {
      if (y < xmax) {
        result = rsqrtpi / y;
      }
    } else {
      ysq = one / (y * y);
      xnum = p[5] * ysq;
      xden = ysq;
      for (int i{0}; i < 4; ++i) {
        xnum = (xnum + p[i]) * ysq;
        xden = (xden + q[i]) * ysq;
      }
      result = ysq * (xnum + p[4]) / (xden + q[4]);
      result = (rsqrtpi - result) / y;
    }
  }
  //  fix up for negative argument, erf, etc.
  if (x < zero) {
    if (x < xneg) {
      result = std::numeric_limits<T>::max();
    } else {
      ysq = trunc(x * sixteen) / sixteen;
      del = (x - ysq) * (x + ysq);
      y = std::exp((ysq * ysq)) * std::exp((del));
      result = (y + y) - result;
    }
  }
  return result;
}

} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_NUMERIC_TEMPLATES_H_
