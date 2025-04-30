/* Test compilation of tgmath macros.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com> and
   Ulrich Drepper <drepper@redhat.com>, 2001.

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

#ifndef HAVE_MAIN
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <tgmath.h>

//#define DEBUG

static void compile_test (void);
static void compile_testf (void);
#if LDBL_MANT_DIG > DBL_MANT_DIG
static void compile_testl (void);
#endif

float fx;
double dx;
long double lx;
const float fy = 1.25;
const double dy = 1.25;
const long double ly = 1.25;
complex float fz;
complex double dz;
complex long double lz;

volatile int count_double;
volatile int count_float;
volatile int count_ldouble;
volatile int count_cdouble;
volatile int count_cfloat;
volatile int count_cldouble;

#define NCALLS     132
#define NCALLS_INT 4
#define NCCALLS    47

static int
do_test (void)
{
  int result = 0;

  count_float = count_double = count_ldouble = 0;
  count_cfloat = count_cdouble = count_cldouble = 0;
  compile_test ();
  if (count_float != 0 || count_cfloat != 0)
    {
      puts ("float function called for double test");
      result = 1;
    }
  if (count_ldouble != 0 || count_cldouble != 0)
    {
      puts ("long double function called for double test");
      result = 1;
    }
  if (count_double < NCALLS + NCALLS_INT)
    {
      printf ("double functions not called often enough (%d)\n",
	      count_double);
      result = 1;
    }
  else if (count_double > NCALLS + NCALLS_INT)
    {
      printf ("double functions called too often (%d)\n",
	      count_double);
      result = 1;
    }
  if (count_cdouble < NCCALLS)
    {
      printf ("double complex functions not called often enough (%d)\n",
	      count_cdouble);
      result = 1;
    }
  else if (count_cdouble > NCCALLS)
    {
      printf ("double complex functions called too often (%d)\n",
	      count_cdouble);
      result = 1;
    }

  count_float = count_double = count_ldouble = 0;
  count_cfloat = count_cdouble = count_cldouble = 0;
  compile_testf ();
  if (count_double != 0 || count_cdouble != 0)
    {
      puts ("double function called for float test");
      result = 1;
    }
  if (count_ldouble != 0 || count_cldouble != 0)
    {
      puts ("long double function called for float test");
      result = 1;
    }
  if (count_float < NCALLS)
    {
      printf ("float functions not called often enough (%d)\n", count_float);
      result = 1;
    }
  else if (count_float > NCALLS)
    {
      printf ("float functions called too often (%d)\n",
	      count_double);
      result = 1;
    }
  if (count_cfloat < NCCALLS)
    {
      printf ("float complex functions not called often enough (%d)\n",
	      count_cfloat);
      result = 1;
    }
  else if (count_cfloat > NCCALLS)
    {
      printf ("float complex functions called too often (%d)\n",
	      count_cfloat);
      result = 1;
    }

#if LDBL_MANT_DIG > DBL_MANT_DIG
  count_float = count_double = count_ldouble = 0;
  count_cfloat = count_cdouble = count_cldouble = 0;
  compile_testl ();
  if (count_float != 0 || count_cfloat != 0)
    {
      puts ("float function called for long double test");
      result = 1;
    }
  if (count_double != 0 || count_cdouble != 0)
    {
      puts ("double function called for long double test");
      result = 1;
    }
  if (count_ldouble < NCALLS)
    {
      printf ("long double functions not called often enough (%d)\n",
	      count_ldouble);
      result = 1;
    }
  else if (count_ldouble > NCALLS)
    {
      printf ("long double functions called too often (%d)\n",
	      count_double);
      result = 1;
    }
  if (count_cldouble < NCCALLS)
    {
      printf ("long double complex functions not called often enough (%d)\n",
	      count_cldouble);
      result = 1;
    }
  else if (count_cldouble > NCCALLS)
    {
      printf ("long double complex functions called too often (%d)\n",
	      count_cldouble);
      result = 1;
    }
#endif

  return result;
}

/* Now generate the three functions.  */
#define HAVE_MAIN

#define F(name) name
#define TYPE double
#define TEST_INT 1
#define x dx
#define y dy
#define z dz
#define count count_double
#define ccount count_cdouble
#include "test-tgmath.c"

#define F(name) name##f
#define TYPE float
#define x fx
#define y fy
#define z fz
#define count count_float
#define ccount count_cfloat
#include "test-tgmath.c"

#if LDBL_MANT_DIG > DBL_MANT_DIG
#define F(name) name##l
#define TYPE long double
#define x lx
#define y ly
#define z lz
#define count count_ldouble
#define ccount count_cldouble
#include "test-tgmath.c"
#endif

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

#else

#ifdef DEBUG
#define P() puts (__FUNCTION__)
#else
#define P()
#endif

static void
F(compile_test) (void)
{
  TYPE a, b, c = 1.0;
  complex TYPE d;
  int i = 2;
  int saved_count;
  long int j;
  long long int k;
  intmax_t m;
  uintmax_t um;

  a = cos (cos (x));
  b = acos (acos (a));
  a = sin (sin (x));
  b = asin (asin (a));
  a = tan (tan (x));
  b = atan (atan (a));
  c = atan2 (atan2 (a, c), atan2 (b, x));
  a = cosh (cosh (x));
  b = acosh (acosh (a));
  a = sinh (sinh (x));
  b = asinh (asinh (a));
  a = tanh (tanh (x));
  b = atanh (atanh (a));
  a = exp (exp (x));
  b = log (log (a));
  a = log10 (log10 (x));
  b = ldexp (ldexp (a, 1), 5);
  a = frexp (frexp (x, &i), &i);
  b = expm1 (expm1 (a));
  a = log1p (log1p (x));
  b = logb (logb (a));
  a = exp2 (exp2 (x));
  b = log2 (log2 (a));
  a = pow (pow (x, a), pow (c, b));
  b = sqrt (sqrt (a));
  a = hypot (hypot (x, b), hypot (c, a));
  b = cbrt (cbrt (a));
  a = ceil (ceil (x));
  b = fabs (fabs (a));
  a = floor (floor (x));
  b = fmod (fmod (a, b), fmod (c, x));
  a = nearbyint (nearbyint (x));
  b = round (round (a));
  c = roundeven (roundeven (a));
  a = trunc (trunc (x));
  b = remquo (remquo (a, b, &i), remquo (c, x, &i), &i);
  j = lrint (x) + lround (a);
  k = llrint (b) + llround (c);
  m = fromfp (a, FP_INT_UPWARD, 2) + fromfpx (b, FP_INT_DOWNWARD, 3);
  um = ufromfp (c, FP_INT_TONEAREST, 4) + ufromfpx (a, FP_INT_TOWARDZERO, 5);
  a = erf (erf (x));
  b = erfc (erfc (a));
  a = tgamma (tgamma (x));
  b = lgamma (lgamma (a));
  a = rint (rint (x));
  b = nextafter (nextafter (a, b), nextafter (c, x));
  a = nextdown (nextdown (a));
  b = nexttoward (nexttoward (x, a), c);
  a = nextup (nextup (a));
  b = remainder (remainder (a, b), remainder (c, x));
  a = scalb (scalb (x, a), (TYPE) (6));
  k = scalbn (a, 7) + scalbln (c, 10l);
  i = ilogb (x);
  j = llogb (x);
  a = fdim (fdim (x, a), fdim (c, b));
  b = fmax (fmax (a, x), fmax (c, b));
  a = fmin (fmin (x, a), fmin (c, b));
  b = fmaxmag (fmaxmag (a, x), fmaxmag (c, b));
  a = fminmag (fminmag (x, a), fminmag (c, b));
  b = fma (sin (a), sin (x), sin (c));

#ifdef TEST_INT
  a = atan2 (i, b);
  b = remquo (i, a, &i);
  c = fma (i, b, i);
  a = pow (i, c);
#endif
  x = a + b + c + i + j + k + m + um;

  saved_count = count;
  if (ccount != 0)
    ccount = -10000;

  d = cos (cos (z));
  z = acos (acos (d));
  d = sin (sin (z));
  z = asin (asin (d));
  d = tan (tan (z));
  z = atan (atan (d));
  d = cosh (cosh (z));
  z = acosh (acosh (d));
  d = sinh (sinh (z));
  z = asinh (asinh (d));
  d = tanh (tanh (z));
  z = atanh (atanh (d));
  d = exp (exp (z));
  z = log (log (d));
  d = sqrt (sqrt (z));
  z = conj (conj (d));
  d = fabs (conj (a));
  z = pow (pow (a, d), pow (b, z));
  d = cproj (cproj (z));
  z += fabs (cproj (a));
  a = carg (carg (z));
  b = creal (creal (d));
  c = cimag (cimag (z));
  x += a + b + c + i + j + k;
  z += d;

  if (saved_count != count)
    count = -10000;

  if (0)
    {
      a = cos (y);
      a = acos (y);
      a = sin (y);
      a = asin (y);
      a = tan (y);
      a = atan (y);
      a = atan2 (y, y);
      a = cosh (y);
      a = acosh (y);
      a = sinh (y);
      a = asinh (y);
      a = tanh (y);
      a = atanh (y);
      a = exp (y);
      a = log (y);
      a = log10 (y);
      a = ldexp (y, 5);
      a = frexp (y, &i);
      a = expm1 (y);
      a = log1p (y);
      a = logb (y);
      a = exp2 (y);
      a = log2 (y);
      a = pow (y, y);
      a = sqrt (y);
      a = hypot (y, y);
      a = cbrt (y);
      a = ceil (y);
      a = fabs (y);
      a = floor (y);
      a = fmod (y, y);
      a = nearbyint (y);
      a = round (y);
      a = roundeven (y);
      a = trunc (y);
      a = remquo (y, y, &i);
      j = lrint (y) + lround (y);
      k = llrint (y) + llround (y);
      m = fromfp (y, FP_INT_UPWARD, 6) + fromfpx (y, FP_INT_DOWNWARD, 7);
      um = (ufromfp (y, FP_INT_TONEAREST, 8)
	    + ufromfpx (y, FP_INT_TOWARDZERO, 9));
      a = erf (y);
      a = erfc (y);
      a = tgamma (y);
      a = lgamma (y);
      a = rint (y);
      a = nextafter (y, y);
      a = nexttoward (y, y);
      a = remainder (y, y);
      a = scalb (y, (const TYPE) (6));
      k = scalbn (y, 7) + scalbln (y, 10l);
      i = ilogb (y);
      j = llogb (y);
      a = fdim (y, y);
      a = fmax (y, y);
      a = fmin (y, y);
      a = fmaxmag (y, y);
      a = fminmag (y, y);
      a = fma (y, y, y);

#ifdef TEST_INT
      a = atan2 (i, y);
      a = remquo (i, y, &i);
      a = fma (i, y, i);
      a = pow (i, y);
#endif

      d = cos ((const complex TYPE) z);
      d = acos ((const complex TYPE) z);
      d = sin ((const complex TYPE) z);
      d = asin ((const complex TYPE) z);
      d = tan ((const complex TYPE) z);
      d = atan ((const complex TYPE) z);
      d = cosh ((const complex TYPE) z);
      d = acosh ((const complex TYPE) z);
      d = sinh ((const complex TYPE) z);
      d = asinh ((const complex TYPE) z);
      d = tanh ((const complex TYPE) z);
      d = atanh ((const complex TYPE) z);
      d = exp ((const complex TYPE) z);
      d = log ((const complex TYPE) z);
      d = sqrt ((const complex TYPE) z);
      d = pow ((const complex TYPE) z, (const complex TYPE) z);
      d = fabs ((const complex TYPE) z);
      d = carg ((const complex TYPE) z);
      d = creal ((const complex TYPE) z);
      d = cimag ((const complex TYPE) z);
      d = conj ((const complex TYPE) z);
      d = cproj ((const complex TYPE) z);
    }
}
#undef x
#undef y
#undef z


TYPE
(F(cos)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(acos)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(sin)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(asin)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(tan)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(atan)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(atan2)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(cosh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(acosh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(sinh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(asinh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(tanh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(atanh)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(exp)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(log)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(log10)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(ldexp)) (TYPE x, int y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(frexp)) (TYPE x, int *y)
{
  ++count;
  P ();
  return x + *y;
}

TYPE
(F(expm1)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(log1p)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(logb)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(exp2)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(log2)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(pow)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(sqrt)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(hypot)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(cbrt)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(ceil)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(fabs)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(floor)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(fmod)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(nearbyint)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(round)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(roundeven)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(trunc)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(remquo)) (TYPE x, TYPE y, int *i)
{
  ++count;
  P ();
  return x + y + *i;
}

long int
(F(lrint)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

long int
(F(lround)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

long long int
(F(llrint)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

long long int
(F(llround)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

intmax_t
(F(fromfp)) (TYPE x, int round, unsigned int width)
{
  ++count;
  P ();
  return x;
}

intmax_t
(F(fromfpx)) (TYPE x, int round, unsigned int width)
{
  ++count;
  P ();
  return x;
}

uintmax_t
(F(ufromfp)) (TYPE x, int round, unsigned int width)
{
  ++count;
  P ();
  return x;
}

uintmax_t
(F(ufromfpx)) (TYPE x, int round, unsigned int width)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(erf)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(erfc)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(tgamma)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(lgamma)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(rint)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(nextafter)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(nextdown)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(nexttoward)) (TYPE x, long double y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(nextup)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(remainder)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(scalb)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(scalbn)) (TYPE x, int y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(scalbln)) (TYPE x, long int y)
{
  ++count;
  P ();
  return x + y;
}

int
(F(ilogb)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

long int
(F(llogb)) (TYPE x)
{
  ++count;
  P ();
  return x;
}

TYPE
(F(fdim)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(fmin)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(fmax)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(fminmag)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(fmaxmag)) (TYPE x, TYPE y)
{
  ++count;
  P ();
  return x + y;
}

TYPE
(F(fma)) (TYPE x, TYPE y, TYPE z)
{
  ++count;
  P ();
  return x + y + z;
}

complex TYPE
(F(cacos)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(casin)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(catan)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(ccos)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(csin)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(ctan)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(cacosh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(casinh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(catanh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(ccosh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(csinh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(ctanh)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(cexp)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(clog)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(csqrt)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(cpow)) (complex TYPE x, complex TYPE y)
{
  ++ccount;
  P ();
  return x + y;
}

TYPE
(F(cabs)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

TYPE
(F(carg)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

TYPE
(F(creal)) (complex TYPE x)
{
  ++ccount;
  P ();
  return __real__ x;
}

TYPE
(F(cimag)) (complex TYPE x)
{
  ++ccount;
  P ();
  return __imag__ x;
}

complex TYPE
(F(conj)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

complex TYPE
(F(cproj)) (complex TYPE x)
{
  ++ccount;
  P ();
  return x;
}

#undef F
#undef TYPE
#undef count
#undef ccount
#undef TEST_INT
#endif
