// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {

  using std::double_t;
  using std::float_t;

  using std::acos;
  using std::acosf;
  using std::acosl;

  using std::asin;
  using std::asinf;
  using std::asinl;

  using std::atan;
  using std::atanf;
  using std::atanl;

  using std::atan2;
  using std::atan2f;
  using std::atan2l;

  using std::cos;
  using std::cosf;
  using std::cosl;

  using std::sin;
  using std::sinf;
  using std::sinl;

  using std::tan;
  using std::tanf;
  using std::tanl;

  using std::acosh;
  using std::acoshf;
  using std::acoshl;

  using std::asinh;
  using std::asinhf;
  using std::asinhl;

  using std::atanh;
  using std::atanhf;
  using std::atanhl;

  using std::cosh;
  using std::coshf;
  using std::coshl;

  using std::sinh;
  using std::sinhf;
  using std::sinhl;

  using std::tanh;
  using std::tanhf;
  using std::tanhl;

  using std::exp;
  using std::expf;
  using std::expl;

  using std::exp2;
  using std::exp2f;
  using std::exp2l;

  using std::expm1;
  using std::expm1f;
  using std::expm1l;

  using std::frexp;
  using std::frexpf;
  using std::frexpl;

  using std::ilogb;
  using std::ilogbf;
  using std::ilogbl;

  using std::ldexp;
  using std::ldexpf;
  using std::ldexpl;

  using std::log;
  using std::logf;
  using std::logl;

  using std::log10;
  using std::log10f;
  using std::log10l;

  using std::log1p;
  using std::log1pf;
  using std::log1pl;

  using std::log2;
  using std::log2f;
  using std::log2l;

  using std::logb;
  using std::logbf;
  using std::logbl;

  using std::modf;
  using std::modff;
  using std::modfl;

  using std::scalbn;
  using std::scalbnf;
  using std::scalbnl;

  using std::scalbln;
  using std::scalblnf;
  using std::scalblnl;

  using std::cbrt;
  using std::cbrtf;
  using std::cbrtl;

  // [c.math.abs], absolute values
  using std::abs;

  using std::fabs;
  using std::fabsf;
  using std::fabsl;

  using std::hypot;
  using std::hypotf;
  using std::hypotl;

  // [c.math.hypot3], three-dimensional hypotenuse

  using std::pow;
  using std::powf;
  using std::powl;

  using std::sqrt;
  using std::sqrtf;
  using std::sqrtl;

  using std::erf;
  using std::erff;
  using std::erfl;

  using std::erfc;
  using std::erfcf;
  using std::erfcl;

  using std::lgamma;
  using std::lgammaf;
  using std::lgammal;

  using std::tgamma;
  using std::tgammaf;
  using std::tgammal;

  using std::ceil;
  using std::ceilf;
  using std::ceill;

  using std::floor;
  using std::floorf;
  using std::floorl;

  using std::nearbyint;
  using std::nearbyintf;
  using std::nearbyintl;

  using std::rint;
  using std::rintf;
  using std::rintl;

  using std::lrint;
  using std::lrintf;
  using std::lrintl;

  using std::llrint;
  using std::llrintf;
  using std::llrintl;

  using std::round;
  using std::roundf;
  using std::roundl;

  using std::lround;
  using std::lroundf;
  using std::lroundl;

  using std::llround;
  using std::llroundf;
  using std::llroundl;

  using std::trunc;
  using std::truncf;
  using std::truncl;

  using std::fmod;
  using std::fmodf;
  using std::fmodl;

  using std::remainder;
  using std::remainderf;
  using std::remainderl;

  using std::remquo;
  using std::remquof;
  using std::remquol;

  using std::copysign;
  using std::copysignf;
  using std::copysignl;

  using std::nan;
  using std::nanf;
  using std::nanl;

  using std::nextafter;
  using std::nextafterf;
  using std::nextafterl;

  using std::nexttoward;
  using std::nexttowardf;
  using std::nexttowardl;

  using std::fdim;
  using std::fdimf;
  using std::fdiml;

  using std::fmax;
  using std::fmaxf;
  using std::fmaxl;

  using std::fmin;
  using std::fminf;
  using std::fminl;

  using std::fma;
  using std::fmaf;
  using std::fmal;

  // [c.math.lerp], linear interpolation
  using std::lerp;

  // [c.math.fpclass], classification / comparison functions
  using std::fpclassify;
  using std::isfinite;
  using std::isgreater;
  using std::isgreaterequal;
  using std::isinf;
  using std::isless;
  using std::islessequal;
  using std::islessgreater;
  using std::isnan;
  using std::isnormal;
  using std::isunordered;
  using std::signbit;

  // [sf.cmath], mathematical special functions
#if 0
  // [sf.cmath.assoc.laguerre], associated Laguerre polynomials
  using std::assoc_laguerre;
  using std::assoc_laguerref;
  using std::assoc_laguerrel;

  // [sf.cmath.assoc.legendre], associated Legendre functions
  using std::assoc_legendre;
  using std::assoc_legendref;
  using std::assoc_legendrel;

  // [sf.cmath.beta], beta function
  using std::beta;
  using std::betaf;
  using std::betal;

  // [sf.cmath.comp.ellint.1], complete elliptic integral of the first kind
  using std::comp_ellint_1;
  using std::comp_ellint_1f;
  using std::comp_ellint_1l;

  // [sf.cmath.comp.ellint.2], complete elliptic integral of the second kind
  using std::comp_ellint_2;
  using std::comp_ellint_2f;
  using std::comp_ellint_2l;

  // [sf.cmath.comp.ellint.3], complete elliptic integral of the third kind
  using std::comp_ellint_3;
  using std::comp_ellint_3f;
  using std::comp_ellint_3l;

  // [sf.cmath.cyl.bessel.i], regular modified cylindrical Bessel functions
  using std::cyl_bessel_i;
  using std::cyl_bessel_if;
  using std::cyl_bessel_il;

  // [sf.cmath.cyl.bessel.j], cylindrical Bessel functions of the first kind
  using std::cyl_bessel_j;
  using std::cyl_bessel_jf;
  using std::cyl_bessel_jl;

  // [sf.cmath.cyl.bessel.k], irregular modified cylindrical Bessel functions
  using std::cyl_bessel_k;
  using std::cyl_bessel_kf;
  using std::cyl_bessel_kl;

  // [sf.cmath.cyl.neumann], cylindrical Neumann functions
  // cylindrical Bessel functions of the second kind
  using std::cyl_neumann;
  using std::cyl_neumannf;
  using std::cyl_neumannl;

  // [sf.cmath.ellint.1], incomplete elliptic integral of the first kind
  using std::ellint_1;
  using std::ellint_1f;
  using std::ellint_1l;

  // [sf.cmath.ellint.2], incomplete elliptic integral of the second kind
  using std::ellint_2;
  using std::ellint_2f;
  using std::ellint_2l;

  // [sf.cmath.ellint.3], incomplete elliptic integral of the third kind
  using std::ellint_3;
  using std::ellint_3f;
  using std::ellint_3l;

  // [sf.cmath.expint], exponential integral
  using std::expint;
  using std::expintf;
  using std::expintl;

  // [sf.cmath.hermite], Hermite polynomials
  using std::hermite;
  using std::hermitef;
  using std::hermitel;

  // [sf.cmath.laguerre], Laguerre polynomials
  using std::laguerre;
  using std::laguerref;
  using std::laguerrel;

  // [sf.cmath.legendre], Legendre polynomials
  using std::legendre;
  using std::legendref;
  using std::legendrel;

  // [sf.cmath.riemann.zeta], Riemann zeta function
  using std::riemann_zeta;
  using std::riemann_zetaf;
  using std::riemann_zetal;

  // [sf.cmath.sph.bessel], spherical Bessel functions of the first kind
  using std::sph_bessel;
  using std::sph_besself;
  using std::sph_bessell;

  // [sf.cmath.sph.legendre], spherical associated Legendre functions
  using std::sph_legendre;
  using std::sph_legendref;
  using std::sph_legendrel;

  // [sf.cmath.sph.neumann], spherical Neumann functions;
  // spherical Bessel functions of the second kind
  using std::sph_neumann;
  using std::sph_neumannf;
  using std::sph_neumannl;
#endif
} // namespace std
