//===-- runtime/Float128Math/math-entries.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
#define FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
#include "terminator.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/entry-names.h"
#include <cfloat>
#include <cmath>
#include <type_traits>

namespace {
using namespace Fortran::runtime;
using F128RetType = CppTypeFor<TypeCategory::Real, 16>;
using I32RetType = CppTypeFor<TypeCategory::Integer, 4>;
using I64RetType = CppTypeFor<TypeCategory::Integer, 8>;
} // namespace

namespace Fortran::runtime {

// Define a class template to gracefully fail, when
// there is no specialized template that implements
// the required function via using the third-party
// implementation.
#define DEFINE_FALLBACK(caller, ret_type) \
  template <bool = false, typename RT = ret_type> struct caller { \
    template <typename... ATs> [[noreturn]] static RT invoke(ATs... args) { \
      Terminator terminator{__FILE__, __LINE__}; \
      terminator.Crash("Float128 variant of '%s' is unsupported", #caller); \
    } \
  };

// Define template specialization that is calling the third-party
// implementation.
//
// Defining the specialization for any target library requires
// adding the generic template via DEFINE_FALLBACK, so that
// a build with another target library that does not define
// the same alias can gracefully fail in runtime.
#define DEFINE_SIMPLE_ALIAS(caller, callee) \
  template <typename RT> struct caller<true, RT> { \
    template <typename... ATs> static RT invoke(ATs... args) { \
      static_assert(std::is_invocable_r_v<RT, \
          decltype(callee(std::declval<ATs>()...))(ATs...), ATs...>); \
      if constexpr (std::is_same_v<RT, void>) { \
        callee(args...); \
      } else { \
        return callee(args...); \
      } \
    } \
  };

// Define fallback callers.
#define DEFINE_FALLBACK_F128(caller) DEFINE_FALLBACK(caller, ::F128RetType)
#define DEFINE_FALLBACK_I32(caller) DEFINE_FALLBACK(caller, ::I32RetType)
#define DEFINE_FALLBACK_I64(caller) DEFINE_FALLBACK(caller, ::I64RetType)

DEFINE_FALLBACK_F128(Abs)
DEFINE_FALLBACK_F128(Acos)
DEFINE_FALLBACK_F128(Acosh)
DEFINE_FALLBACK_F128(Asin)
DEFINE_FALLBACK_F128(Asinh)
DEFINE_FALLBACK_F128(Atan)
DEFINE_FALLBACK_F128(Atan2)
DEFINE_FALLBACK_F128(Atanh)
DEFINE_FALLBACK_F128(Ceil)
DEFINE_FALLBACK_F128(Cos)
DEFINE_FALLBACK_F128(Cosh)
DEFINE_FALLBACK_F128(Erf)
DEFINE_FALLBACK_F128(Erfc)
DEFINE_FALLBACK_F128(Exp)
DEFINE_FALLBACK_F128(Floor)
DEFINE_FALLBACK_F128(Fma)
DEFINE_FALLBACK_F128(Frexp)
DEFINE_FALLBACK_F128(Hypot)
DEFINE_FALLBACK_I32(Ilogb)
DEFINE_FALLBACK_I32(Isinf)
DEFINE_FALLBACK_I32(Isnan)
DEFINE_FALLBACK_F128(J0)
DEFINE_FALLBACK_F128(J1)
DEFINE_FALLBACK_F128(Jn)
DEFINE_FALLBACK_F128(Ldexp)
DEFINE_FALLBACK_F128(Lgamma)
DEFINE_FALLBACK_I64(Llround)
DEFINE_FALLBACK_F128(Log)
DEFINE_FALLBACK_F128(Log10)
DEFINE_FALLBACK_I32(Lround)
DEFINE_FALLBACK_F128(Nextafter)
DEFINE_FALLBACK_F128(Pow)
DEFINE_FALLBACK_F128(Qnan)
DEFINE_FALLBACK_F128(Round)
DEFINE_FALLBACK_F128(Sin)
DEFINE_FALLBACK_F128(Sinh)
DEFINE_FALLBACK_F128(Sqrt)
DEFINE_FALLBACK_F128(Tan)
DEFINE_FALLBACK_F128(Tanh)
DEFINE_FALLBACK_F128(Tgamma)
DEFINE_FALLBACK_F128(Trunc)
DEFINE_FALLBACK_F128(Y0)
DEFINE_FALLBACK_F128(Y1)
DEFINE_FALLBACK_F128(Yn)

#if HAS_QUADMATHLIB
// Define wrapper callers for libquadmath.
#include "quadmath.h"
DEFINE_SIMPLE_ALIAS(Abs, fabsq)
DEFINE_SIMPLE_ALIAS(Acos, acosq)
DEFINE_SIMPLE_ALIAS(Acosh, acoshq)
DEFINE_SIMPLE_ALIAS(Asin, asinq)
DEFINE_SIMPLE_ALIAS(Asinh, asinhq)
DEFINE_SIMPLE_ALIAS(Atan, atanq)
DEFINE_SIMPLE_ALIAS(Atan2, atan2q)
DEFINE_SIMPLE_ALIAS(Atanh, atanhq)
DEFINE_SIMPLE_ALIAS(Ceil, ceilq)
DEFINE_SIMPLE_ALIAS(Cos, cosq)
DEFINE_SIMPLE_ALIAS(Cosh, coshq)
DEFINE_SIMPLE_ALIAS(Erf, erfq)
DEFINE_SIMPLE_ALIAS(Erfc, erfcq)
DEFINE_SIMPLE_ALIAS(Exp, expq)
DEFINE_SIMPLE_ALIAS(Floor, floorq)
DEFINE_SIMPLE_ALIAS(Fma, fmaq)
DEFINE_SIMPLE_ALIAS(Frexp, frexpq)
DEFINE_SIMPLE_ALIAS(Hypot, hypotq)
DEFINE_SIMPLE_ALIAS(Ilogb, ilogbq)
DEFINE_SIMPLE_ALIAS(Isinf, isinfq)
DEFINE_SIMPLE_ALIAS(Isnan, isnanq)
DEFINE_SIMPLE_ALIAS(J0, j0q)
DEFINE_SIMPLE_ALIAS(J1, j1q)
DEFINE_SIMPLE_ALIAS(Jn, jnq)
DEFINE_SIMPLE_ALIAS(Ldexp, ldexpq)
DEFINE_SIMPLE_ALIAS(Lgamma, lgammaq)
DEFINE_SIMPLE_ALIAS(Llround, llroundq)
DEFINE_SIMPLE_ALIAS(Log, logq)
DEFINE_SIMPLE_ALIAS(Log10, log10q)
DEFINE_SIMPLE_ALIAS(Lround, lroundq)
DEFINE_SIMPLE_ALIAS(Nextafter, nextafterq)
DEFINE_SIMPLE_ALIAS(Pow, powq)
DEFINE_SIMPLE_ALIAS(Round, roundq)
DEFINE_SIMPLE_ALIAS(Sin, sinq)
DEFINE_SIMPLE_ALIAS(Sinh, sinhq)
DEFINE_SIMPLE_ALIAS(Sqrt, sqrtq)
DEFINE_SIMPLE_ALIAS(Tan, tanq)
DEFINE_SIMPLE_ALIAS(Tanh, tanhq)
DEFINE_SIMPLE_ALIAS(Tgamma, tgammaq)
DEFINE_SIMPLE_ALIAS(Trunc, truncq)
DEFINE_SIMPLE_ALIAS(Y0, y0q)
DEFINE_SIMPLE_ALIAS(Y1, y1q)
DEFINE_SIMPLE_ALIAS(Yn, ynq)

// Use cmath INFINITY/NAN definition. Rely on C implicit conversions.
#define F128_RT_INFINITY (INFINITY)
#define F128_RT_QNAN (NAN)
#elif LDBL_MANT_DIG == 113
// Define wrapper callers for libm.
#include <limits>

// Use STD math functions. They provide IEEE-754 128-bit float
// support either via 'long double' or __float128.
// The Bessel's functions are not present in STD namespace.
DEFINE_SIMPLE_ALIAS(Abs, std::abs)
DEFINE_SIMPLE_ALIAS(Acos, std::acos)
DEFINE_SIMPLE_ALIAS(Acosh, std::acosh)
DEFINE_SIMPLE_ALIAS(Asin, std::asin)
DEFINE_SIMPLE_ALIAS(Asinh, std::asinh)
DEFINE_SIMPLE_ALIAS(Atan, std::atan)
DEFINE_SIMPLE_ALIAS(Atan2, std::atan2)
DEFINE_SIMPLE_ALIAS(Atanh, std::atanh)
DEFINE_SIMPLE_ALIAS(Ceil, std::ceil)
DEFINE_SIMPLE_ALIAS(Cos, std::cos)
DEFINE_SIMPLE_ALIAS(Cosh, std::cosh)
DEFINE_SIMPLE_ALIAS(Erf, std::erf)
DEFINE_SIMPLE_ALIAS(Erfc, std::erfc)
DEFINE_SIMPLE_ALIAS(Exp, std::exp)
DEFINE_SIMPLE_ALIAS(Floor, std::floor)
DEFINE_SIMPLE_ALIAS(Fma, std::fma)
DEFINE_SIMPLE_ALIAS(Frexp, std::frexp)
DEFINE_SIMPLE_ALIAS(Hypot, std::hypot)
DEFINE_SIMPLE_ALIAS(Ilogb, std::ilogb)
DEFINE_SIMPLE_ALIAS(Isinf, std::isinf)
DEFINE_SIMPLE_ALIAS(Isnan, std::isnan)
DEFINE_SIMPLE_ALIAS(J0, j0l)
DEFINE_SIMPLE_ALIAS(J1, j1l)
DEFINE_SIMPLE_ALIAS(Jn, jnl)
DEFINE_SIMPLE_ALIAS(Ldexp, std::ldexp)
DEFINE_SIMPLE_ALIAS(Lgamma, std::lgamma)
DEFINE_SIMPLE_ALIAS(Llround, std::llround)
DEFINE_SIMPLE_ALIAS(Log, std::log)
DEFINE_SIMPLE_ALIAS(Log10, std::log10)
DEFINE_SIMPLE_ALIAS(Lround, std::lround)
DEFINE_SIMPLE_ALIAS(Nextafter, std::nextafter)
DEFINE_SIMPLE_ALIAS(Pow, std::pow)
DEFINE_SIMPLE_ALIAS(Round, std::round)
DEFINE_SIMPLE_ALIAS(Sin, std::sin)
DEFINE_SIMPLE_ALIAS(Sinh, std::sinh)
DEFINE_SIMPLE_ALIAS(Sqrt, std::sqrt)
DEFINE_SIMPLE_ALIAS(Tan, std::tan)
DEFINE_SIMPLE_ALIAS(Tanh, std::tanh)
DEFINE_SIMPLE_ALIAS(Tgamma, std::tgamma)
DEFINE_SIMPLE_ALIAS(Trunc, std::trunc)
DEFINE_SIMPLE_ALIAS(Y0, y0l)
DEFINE_SIMPLE_ALIAS(Y1, y1l)
DEFINE_SIMPLE_ALIAS(Yn, ynl)

// Use numeric_limits to produce infinity of the right type.
#define F128_RT_INFINITY \
  (std::numeric_limits<CppTypeFor<TypeCategory::Real, 16>>::infinity())
#define F128_RT_QNAN \
  (std::numeric_limits<CppTypeFor<TypeCategory::Real, 16>>::quiet_NaN())
#elif HAS_LIBMF128
// We can use __float128 versions of libm functions.
// __STDC_WANT_IEC_60559_TYPES_EXT__ needs to be defined
// before including cmath to enable the *f128 prototypes.
#error "Float128Math build with glibc>=2.26 is unsupported yet"
#endif

} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
