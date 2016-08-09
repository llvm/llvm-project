/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#define ATTR __attribute__((always_inline))

#define FUNC1(name, ocml_name, T) \
  ATTR T name(T x) \
  { \
    return ocml_name(x); \
  }

#define FUNC1A(name) \
  FUNC1(__hc_##name, __ocml_##name##_f32, float) \
  FUNC1(__hc_##name##_double, __ocml_##name##_f64, double)

#define FUNC2(name, ocml_name, T) \
  ATTR T name(T x, T y) \
  { \
    return ocml_name(x, y); \
  }

#define FUNC2A(name) \
  FUNC2(__hc_##name, __ocml_##name##_f32, float) \
  FUNC2(__hc_##name##_double, __ocml_##name##_f64, double)

FUNC1A(acos)
FUNC1A(acosh)
FUNC1A(asin)
FUNC1A(asinh)
FUNC1A(atan)
FUNC1A(atanh)
FUNC2A(atan2)
FUNC1A(cbrt)
FUNC1A(ceil)
FUNC2A(copysign)
FUNC1A(cos)
FUNC1A(cosh)
FUNC1A(cospi)
FUNC1A(erf)
FUNC1A(erfc)
FUNC1A(exp)
FUNC1A(exp2)
FUNC1A(exp10)
FUNC1A(expm1)
FUNC1A(fabs)
FUNC2A(fdim)
FUNC1A(floor)
//FUNC3A(fma)
FUNC2A(fmax)
FUNC2A(fmin)
FUNC2A(fmod)
// frexp
// frexp_global
// frexp_local
FUNC2A(hypot)
// ilogb
// isfinite
// isinf
// isnan
// isnormal
// lgammaf
// lgamma
FUNC1A(log)
FUNC1A(log10)
FUNC1A(log2)
FUNC1A(log1p)
FUNC1A(logb)
// modff
// modf
// nan
FUNC1A(nearbyint)
FUNC2A(nextafter)
FUNC2A(pow)
FUNC2A(remainder)
// remquof
// remquo
FUNC1A(round)
FUNC1A(rsqrt)
FUNC1A(sinpi)
// ldexp
// signbit
FUNC1A(sin)
// sincos
FUNC1A(sinh)
FUNC1A(sqrt)
FUNC1A(tgamma)
FUNC1A(tan)
FUNC1A(tanh)
FUNC1A(tanpi)
FUNC1A(trunc)
