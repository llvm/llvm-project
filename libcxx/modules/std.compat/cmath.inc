// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::double_t;
  using ::float_t;

  using ::acos;
  using ::acosf;
  using ::acosl;

  using ::asin;
  using ::asinf;
  using ::asinl;

  using ::atan;
  using ::atanf;
  using ::atanl;

  using ::atan2;
  using ::atan2f;
  using ::atan2l;

  using ::cos;
  using ::cosf;
  using ::cosl;

  using ::sin;
  using ::sinf;
  using ::sinl;

  using ::tan;
  using ::tanf;
  using ::tanl;

  using ::acosh;
  using ::acoshf;
  using ::acoshl;

  using ::asinh;
  using ::asinhf;
  using ::asinhl;

  using ::atanh;
  using ::atanhf;
  using ::atanhl;

  using ::cosh;
  using ::coshf;
  using ::coshl;

  using ::sinh;
  using ::sinhf;
  using ::sinhl;

  using ::tanh;
  using ::tanhf;
  using ::tanhl;

  using ::exp;
  using ::expf;
  using ::expl;

  using ::exp2;
  using ::exp2f;
  using ::exp2l;

  using ::expm1;
  using ::expm1f;
  using ::expm1l;

  using ::frexp;
  using ::frexpf;
  using ::frexpl;

  using ::ilogb;
  using ::ilogbf;
  using ::ilogbl;

  using ::ldexp;
  using ::ldexpf;
  using ::ldexpl;

  using ::log;
  using ::logf;
  using ::logl;

  using ::log10;
  using ::log10f;
  using ::log10l;

  using ::log1p;
  using ::log1pf;
  using ::log1pl;

  using ::log2;
  using ::log2f;
  using ::log2l;

  using ::logb;
  using ::logbf;
  using ::logbl;

  using ::modf;
  using ::modff;
  using ::modfl;

  using ::scalbn;
  using ::scalbnf;
  using ::scalbnl;

  using ::scalbln;
  using ::scalblnf;
  using ::scalblnl;

  using ::cbrt;
  using ::cbrtf;
  using ::cbrtl;

  // [c.math.abs], absolute values
  using ::abs;

  using ::fabs;
  using ::fabsf;
  using ::fabsl;

  using ::hypot;
  using ::hypotf;
  using ::hypotl;

  // [c.math.hypot3], three-dimensional hypotenuse

  using ::pow;
  using ::powf;
  using ::powl;

  using ::sqrt;
  using ::sqrtf;
  using ::sqrtl;

  using ::erf;
  using ::erff;
  using ::erfl;

  using ::erfc;
  using ::erfcf;
  using ::erfcl;

  using ::lgamma;
  using ::lgammaf;
  using ::lgammal;

  using ::tgamma;
  using ::tgammaf;
  using ::tgammal;

  using ::ceil;
  using ::ceilf;
  using ::ceill;

  using ::floor;
  using ::floorf;
  using ::floorl;

  using ::nearbyint;
  using ::nearbyintf;
  using ::nearbyintl;

  using ::rint;
  using ::rintf;
  using ::rintl;

  using ::lrint;
  using ::lrintf;
  using ::lrintl;

  using ::llrint;
  using ::llrintf;
  using ::llrintl;

  using ::round;
  using ::roundf;
  using ::roundl;

  using ::lround;
  using ::lroundf;
  using ::lroundl;

  using ::llround;
  using ::llroundf;
  using ::llroundl;

  using ::trunc;
  using ::truncf;
  using ::truncl;

  using ::fmod;
  using ::fmodf;
  using ::fmodl;

  using ::remainder;
  using ::remainderf;
  using ::remainderl;

  using ::remquo;
  using ::remquof;
  using ::remquol;

  using ::copysign;
  using ::copysignf;
  using ::copysignl;

  using ::nan;
  using ::nanf;
  using ::nanl;

  using ::nextafter;
  using ::nextafterf;
  using ::nextafterl;

  using ::nexttoward;
  using ::nexttowardf;
  using ::nexttowardl;

  using ::fdim;
  using ::fdimf;
  using ::fdiml;

  using ::fmax;
  using ::fmaxf;
  using ::fmaxl;

  using ::fmin;
  using ::fminf;
  using ::fminl;

  using ::fma;
  using ::fmaf;
  using ::fmal;

  // [c.math.lerp], linear interpolation
  // [support.c.headers.other]/1
  // ...  placed within the global namespace scope, except for the functions
  // described in [sf.cmath], the std::lerp function overloads ([c.math.lerp])
  // ...

  // [c.math.fpclass], classification / comparison functions
  using ::fpclassify;
  using ::isfinite;
  using ::isgreater;
  using ::isgreaterequal;
  using ::isinf;
  using ::isless;
  using ::islessequal;
  using ::islessgreater;
  using ::isnan;
  using ::isnormal;
  using ::isunordered;
  using ::signbit;

  // [sf.cmath], mathematical special functions
} // export
