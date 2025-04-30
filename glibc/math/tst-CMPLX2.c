/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Marek Polacek <polacek@redhat.com>, 2012.

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

/* Adapted from gcc.dg/torture/builtin-complex-1.c test from GCC
   testsuite written by Joseph S. Myers.  */

#include <complex.h>

static int result;

#define COMPARE_BODY(A, B, TYPE, COPYSIGN)				\
  do {									\
    TYPE s1 = COPYSIGN ((TYPE) 1.0, A);					\
    TYPE s2 = COPYSIGN ((TYPE) 1.0, B);					\
    if (s1 != s2)							\
      result |= 1;							\
    if ((__builtin_isnan (A) != 0) != (__builtin_isnan (B) != 0))	\
      result |= 1;							\
    if ((A != B) != (__builtin_isnan (A) != 0))				\
      result |= 1;							\
  } while (0)

#ifdef CMPLX

static void
comparef (float a, float b)
{
  COMPARE_BODY (a, b, float, __builtin_copysignf);
}

static void
compare (double a, double b)
{
  COMPARE_BODY (a, b, double, __builtin_copysign);
}

static void
comparel (long double a, long double b)
{
  COMPARE_BODY (a, b, long double, __builtin_copysignl);
}

static void
comparecf (_Complex float a, float r, float i)
{
  comparef (__real__ a, r);
  comparef (__imag__ a, i);
}

static void
comparec (_Complex double a, double r, double i)
{
  compare (__real__ a, r);
  compare (__imag__ a, i);
}

static void
comparecl (_Complex long double a, long double r, long double i)
{
  comparel (__real__ a, r);
  comparel (__imag__ a, i);
}

#define VERIFY(A, B, TYPE, COMPARE, CL)			\
  do {							\
    TYPE a = A;						\
    TYPE b = B;						\
    _Complex TYPE cr = CL (a, b);			\
    static _Complex TYPE cs = CL (A, B);		\
    COMPARE (cr, A, B);					\
    COMPARE (cs, A, B);					\
  } while (0)

#define ALL_CHECKS(PZ, NZ, NAN, INF, TYPE, COMPARE, CL)	\
  do {							\
    VERIFY (PZ, PZ, TYPE, COMPARE, CL);			\
    VERIFY (PZ, NZ, TYPE, COMPARE, CL);			\
    VERIFY (PZ, NAN, TYPE, COMPARE, CL);		\
    VERIFY (PZ, INF, TYPE, COMPARE, CL);		\
    VERIFY (NZ, PZ, TYPE, COMPARE, CL);			\
    VERIFY (NZ, NZ, TYPE, COMPARE, CL);			\
    VERIFY (NZ, NAN, TYPE, COMPARE, CL);		\
    VERIFY (NZ, INF, TYPE, COMPARE, CL);		\
    VERIFY (NAN, PZ, TYPE, COMPARE, CL);		\
    VERIFY (NAN, NZ, TYPE, COMPARE, CL);		\
    VERIFY (NAN, NAN, TYPE, COMPARE, CL);		\
    VERIFY (NAN, INF, TYPE, COMPARE, CL);		\
    VERIFY (INF, PZ, TYPE, COMPARE,CL);			\
    VERIFY (INF, NZ, TYPE, COMPARE, CL);		\
    VERIFY (INF, NAN, TYPE, COMPARE, CL);		\
    VERIFY (INF, INF, TYPE, COMPARE, CL);		\
  } while (0)

static void
check_float (void)
{
  ALL_CHECKS (0.0f, -0.0f, __builtin_nanf (""), __builtin_inff (),
	      float, comparecf, CMPLXF);
}

static void
check_double (void)
{
  ALL_CHECKS (0.0, -0.0, __builtin_nan (""), __builtin_inf (),
	      double, comparec, CMPLX);
}

static void
check_long_double (void)
{
  ALL_CHECKS (0.0l, -0.0l, __builtin_nanl (""), __builtin_infl (),
	      long double, comparecl, CMPLXL);
}
#endif

static int
do_test (void)
{
#ifdef CMPLX
  check_float ();
  check_double ();
  check_long_double ();
#endif

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
