/* Test C99 math functions are available in C++11 without _GNU_SOURCE.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#undef _GNU_SOURCE
#undef _DEFAULT_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_SOURCE
#undef _POSIX_C_SOURCE
// __STRICT_ANSI__ gets defined by -std=c++11 in CFLAGS
#include <math.h>
#include <stdio.h>

static int
do_test (void)
{
#ifdef _GNU_SOURCE
  printf ("FAIL: _GNU_SOURCE is defined.\n");
  return 1;
#endif

#if __cplusplus >= 201103L
  /* Verify that C11 math functions and types are defined for C++11,
     without _GNU_SOURCE being defined. [BZ #21326] */
  (void) FP_INFINITE;
  (void) FP_NAN;
  (void) FP_NORMAL;
  (void) FP_SUBNORMAL;
  (void) FP_ZERO;
  double_t d = 1.0;
  (void) d;
  float_t f = 1.0f;
  (void) f;
  (void) acosh;
  (void) acoshf;
  (void) acoshl;
  (void) asinh;
  (void) asinhf;
  (void) asinhl;
  (void) atanh;
  (void) atanhf;
  (void) atanhl;
  (void) cbrt;
  (void) cbrtf;
  (void) cbrtl;
  (void) copysign;
  (void) copysignf;
  (void) copysignl;
  (void) erf;
  (void) erff;
  (void) erfl;
  (void) erfc;
  (void) erfcf;
  (void) erfcl;
  (void) exp2;
  (void) exp2f;
  (void) exp2l;
  (void) expm1;
  (void) expm1f;
  (void) expm1l;
  (void) fdim;
  (void) fdimf;
  (void) fdiml;
  (void) fma;
  (void) fmaf;
  (void) fmal;
  (void) fmax;
  (void) fmaxf;
  (void) fmaxl;
  (void) fmin;
  (void) fminf;
  (void) fminl;
  (void) hypot;
  (void) hypotf;
  (void) hypotl;
  (void) ilogb;
  (void) ilogbf;
  (void) ilogbl;
  (void) lgamma;
  (void) lgammaf;
  (void) lgammal;
  (void) llrint;
  (void) llrintf;
  (void) llrintl;
  (void) llround;
  (void) llroundf;
  (void) llroundl;
  (void) log1p;
  (void) log1pf;
  (void) log1pl;
  (void) log2;
  (void) log2f;
  (void) log2l;
  (void) logb;
  (void) logbf;
  (void) logbl;
  (void) lrint;
  (void) lrintf;
  (void) lrintl;
  (void) lround;
  (void) lroundf;
  (void) lroundl;
  (void) nan;
  (void) nanf;
  (void) nanl;
  (void) nearbyint;
  (void) nearbyintf;
  (void) nearbyintl;
  (void) nextafter;
  (void) nextafterf;
  (void) nextafterl;
  (void) nexttoward;
  (void) nexttowardf;
  (void) nexttowardl;
  (void) remainder;
  (void) remainderf;
  (void) remainderl;
  (void) remquo;
  (void) remquof;
  (void) remquol;
  (void) rint;
  (void) rintf;
  (void) rintl;
  (void) round;
  (void) roundf;
  (void) roundl;
  (void) scalbln;
  (void) scalblnf;
  (void) scalblnl;
  (void) scalbn;
  (void) scalbnf;
  (void) scalbnl;
  (void) tgamma;
  (void) tgammaf;
  (void) tgammal;
  (void) trunc;
  (void) truncf;
  (void) truncl;
  printf ("PASS: C11 math functions present in C++11 without _GNU_SOURCE.\n");
#else
  printf ("UNSUPPORTED: C++11 not enabled.\n");
#endif
  return 0;
}

#include <support/test-driver.c>
