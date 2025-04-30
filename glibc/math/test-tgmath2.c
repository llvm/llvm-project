/* Test compilation of tgmath macros.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2007.

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
#include <complex.h>
#include <stdio.h>
#include <string.h>
#include <tgmath.h>

//#define DEBUG

typedef complex float cfloat;
typedef complex double cdouble;
#if LDBL_MANT_DIG > DBL_MANT_DIG
typedef long double ldouble;
typedef complex long double cldouble;
#else
typedef double ldouble;
typedef complex double cldouble;
#endif

float vfloat1, vfloat2, vfloat3;
double vdouble1, vdouble2, vdouble3;
ldouble vldouble1, vldouble2, vldouble3;
cfloat vcfloat1, vcfloat2, vcfloat3;
cdouble vcdouble1, vcdouble2, vcdouble3;
cldouble vcldouble1, vcldouble2, vcldouble4;
int vint1, vint2, vint3;
long int vlong1, vlong2, vlong3;
long long int vllong1, vllong2, vllong3;
const float Vfloat1 = 1, Vfloat2 = 2, Vfloat3 = 3;
const double Vdouble1 = 1, Vdouble2 = 2, Vdouble3 = 3;
const ldouble Vldouble1 = 1, Vldouble2 = 2, Vldouble3 = 3;
const cfloat Vcfloat1 = 1, Vcfloat2 = 2, Vcfloat3 = 3;
const cdouble Vcdouble1 = 1, Vcdouble2 = 2, Vcdouble3 = 3;
const cldouble Vcldouble1 = 1, Vcldouble2 = 2, Vcldouble4 = 3;
const int Vint1 = 1, Vint2 = 2, Vint3 = 3;
const long int Vlong1 = 1, Vlong2 = 2, Vlong3 = 3;
const long long int Vllong1 = 1, Vllong2 = 2, Vllong3 = 3;
enum
  {
    Tfloat = 0,
    Tcfloat,
    Tdouble,
    Tcdouble,
#if LDBL_MANT_DIG > DBL_MANT_DIG
    Tldouble,
    Tcldouble,
#else
    Tldouble = Tdouble,
    Tcldouble = Tcdouble,
#endif
    Tlast
  };
enum
  {
    C_cos = 0,
    C_fabs,
    C_cabs,
    C_conj,
    C_expm1,
    C_lrint,
    C_ldexp,
    C_atan2,
    C_remquo,
    C_pow,
    C_fma,
    C_last
  };
int count;
int counts[Tlast][C_last];

#define FAIL(str) \
  do								\
    {								\
      printf ("%s failure on line %d\n", (str), __LINE__);	\
      result = 1;						\
    }								\
  while (0)
#define TEST_TYPE_ONLY(expr, rettype) \
  do								\
    {								\
      __typeof__ (expr) texpr = 0;				\
      __typeof__ (rettype) ttype = 0, *ptype;			\
      if (sizeof (expr) != sizeof (rettype))			\
	FAIL ("type");						\
      if (__alignof__ (expr) != __alignof__ (rettype))		\
	FAIL ("type");						\
      __asm ("" : "=r" (ptype) : "0" (&ttype), "r" (&texpr));	\
      if (&texpr == ptype)					\
	FAIL ("type");						\
    }								\
  while (0)
#define TEST2(expr, type, rettype, fn) \
  do								\
    {								\
      __typeof__ (expr) texpr = 0;				\
      TEST_TYPE_ONLY (expr, rettype);				\
      if (count != 0)						\
	FAIL ("internal error");				\
      if (counts[T##type][C_##fn] != 0)				\
	FAIL ("internal error");				\
      texpr = expr;						\
      __asm __volatile ("" : : "r" (&texpr));			\
      if (count != 1 || counts[T##type][C_##fn] != 1)		\
	{							\
	  FAIL ("wrong function called, "#fn" ("#type")");	\
	  memset (counts, 0, sizeof (counts));			\
	}							\
      count = 0;						\
      counts[T##type][C_##fn] = 0;				\
    }								\
  while (0)
#define TEST(expr, type, fn) TEST2(expr, type, type, fn)

int
test_cos (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  TEST (cos (vfloat1), float, cos);
  TEST (cos (vdouble1), double, cos);
  TEST (cos (vldouble1), ldouble, cos);
  TEST (cos (vint1), double, cos);
  TEST (cos (vllong1), double, cos);
  TEST (cos (vcfloat1), cfloat, cos);
  TEST (cos (vcdouble1), cdouble, cos);
  TEST (cos (vcldouble1), cldouble, cos);
  TEST (cos (Vfloat1), float, cos);
  TEST (cos (Vdouble1), double, cos);
  TEST (cos (Vldouble1), ldouble, cos);
  TEST (cos (Vint1), double, cos);
  TEST (cos (Vllong1), double, cos);
  TEST (cos (Vcfloat1), cfloat, cos);
  TEST (cos (Vcdouble1), cdouble, cos);
  TEST (cos (Vcldouble1), cldouble, cos);

  return result;
}

int
test_fabs (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  TEST (fabs (vfloat1), float, fabs);
  TEST (fabs (vdouble1), double, fabs);
  TEST (fabs (vldouble1), ldouble, fabs);
  TEST (fabs (vint1), double, fabs);
  TEST (fabs (vllong1), double, fabs);
  TEST (fabs (vcfloat1), float, cabs);
  TEST (fabs (vcdouble1), double, cabs);
  TEST (fabs (vcldouble1), ldouble, cabs);
  TEST (fabs (Vfloat1), float, fabs);
  TEST (fabs (Vdouble1), double, fabs);
  TEST (fabs (Vldouble2), ldouble, fabs);
#ifndef __OPTIMIZE__
  /* GCC is too smart to optimize these out.  */
  TEST (fabs (Vint1), double, fabs);
  TEST (fabs (Vllong1), double, fabs);
#else
  TEST_TYPE_ONLY (fabs (vllong1), double);
  TEST_TYPE_ONLY (fabs (vllong1), double);
#endif
  TEST (fabs (Vint4), double, fabs);
  TEST (fabs (Vllong4), double, fabs);
  TEST (fabs (Vcfloat1), float, cabs);
  TEST (fabs (Vcdouble1), double, cabs);
  TEST (fabs (Vcldouble1), ldouble, cabs);

  return result;
}

int
test_conj (const int Vint4, const long long int Vllong4)
{
  int result = 0;
  TEST (conj (vfloat1), cfloat, conj);
  TEST (conj (vdouble1), cdouble, conj);
  TEST (conj (vldouble1), cldouble, conj);
  TEST (conj (vint1), cdouble, conj);
  TEST (conj (vllong1), cdouble, conj);
  TEST (conj (vcfloat1), cfloat, conj);
  TEST (conj (vcdouble1), cdouble, conj);
  TEST (conj (vcldouble1), cldouble, conj);
  TEST (conj (Vfloat1), cfloat, conj);
  TEST (conj (Vdouble1), cdouble, conj);
  TEST (conj (Vldouble1), cldouble, conj);
  TEST (conj (Vint1), cdouble, conj);
  TEST (conj (Vllong1), cdouble, conj);
  TEST (conj (Vcfloat1), cfloat, conj);
  TEST (conj (Vcdouble1), cdouble, conj);
  TEST (conj (Vcldouble1), cldouble, conj);

  return result;
}

int
test_expm1 (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  TEST (expm1 (vfloat1), float, expm1);
  TEST (expm1 (vdouble1), double, expm1);
  TEST (expm1 (vldouble1), ldouble, expm1);
  TEST (expm1 (vint1), double, expm1);
  TEST (expm1 (vllong1), double, expm1);
  TEST (expm1 (Vfloat1), float, expm1);
  TEST (expm1 (Vdouble1), double, expm1);
  TEST (expm1 (Vldouble1), ldouble, expm1);
  TEST (expm1 (Vint1), double, expm1);
  TEST (expm1 (Vllong1), double, expm1);

  return result;
}

int
test_lrint (const int Vint4, const long long int Vllong4)
{
  int result = 0;
  TEST2 (lrint (vfloat1), float, long int, lrint);
  TEST2 (lrint (vdouble1), double, long int, lrint);
  TEST2 (lrint (vldouble1), ldouble, long int, lrint);
  TEST2 (lrint (vint1), double, long int, lrint);
  TEST2 (lrint (vllong1), double, long int, lrint);
  TEST2 (lrint (Vfloat1), float, long int, lrint);
  TEST2 (lrint (Vdouble1), double, long int, lrint);
  TEST2 (lrint (Vldouble1), ldouble, long int, lrint);
  TEST2 (lrint (Vint1), double, long int, lrint);
  TEST2 (lrint (Vllong1), double, long int, lrint);

  return result;
}

int
test_ldexp (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  TEST (ldexp (vfloat1, 6), float, ldexp);
  TEST (ldexp (vdouble1, 6), double, ldexp);
  TEST (ldexp (vldouble1, 6), ldouble, ldexp);
  TEST (ldexp (vint1, 6), double, ldexp);
  TEST (ldexp (vllong1, 6), double, ldexp);
  TEST (ldexp (Vfloat1, 6), float, ldexp);
  TEST (ldexp (Vdouble1, 6), double, ldexp);
  TEST (ldexp (Vldouble1, 6), ldouble, ldexp);
  TEST (ldexp (Vint1, 6), double, ldexp);
  TEST (ldexp (Vllong1, 6), double, ldexp);

  return result;
}

#define FIRST(x, y) (y, x)
#define SECOND(x, y) (x, y)
#define NON_LDBL_TEST(fn, argm, arg, type, fnt) \
  TEST (fn argm (arg, vfloat1), type, fnt); \
  TEST (fn argm (arg, vdouble1), type, fnt); \
  TEST (fn argm (arg, vint1), type, fnt); \
  TEST (fn argm (arg, vllong1), type, fnt); \
  TEST (fn argm (arg, Vfloat1), type, fnt); \
  TEST (fn argm (arg, Vdouble1), type, fnt); \
  TEST (fn argm (arg, Vint1), type, fnt); \
  TEST (fn argm (arg, Vllong1), type, fnt);
#define NON_LDBL_CTEST(fn, argm, arg, type, fnt) \
  NON_LDBL_TEST(fn, argm, arg, type, fnt); \
  TEST (fn argm (arg, vcfloat1), type, fnt); \
  TEST (fn argm (arg, vcdouble1), type, fnt); \
  TEST (fn argm (arg, Vcfloat1), type, fnt); \
  TEST (fn argm (arg, Vcdouble1), type, fnt);
#define BINARY_TEST(fn, fnt) \
  TEST (fn (vfloat1, vfloat2), float, fnt); \
  TEST (fn (Vfloat1, vfloat2), float, fnt); \
  TEST (fn (vfloat1, Vfloat2), float, fnt); \
  TEST (fn (Vfloat1, Vfloat2), float, fnt); \
  TEST (fn (vldouble1, vldouble2), ldouble, fnt); \
  TEST (fn (Vldouble1, vldouble2), ldouble, fnt); \
  TEST (fn (vldouble1, Vldouble2), ldouble, fnt); \
  TEST (fn (Vldouble1, Vldouble2), ldouble, fnt); \
  NON_LDBL_TEST (fn, FIRST, vldouble2, ldouble, fnt); \
  NON_LDBL_TEST (fn, SECOND, vldouble2, ldouble, fnt); \
  NON_LDBL_TEST (fn, FIRST, Vldouble2, ldouble, fnt); \
  NON_LDBL_TEST (fn, SECOND, Vldouble2, ldouble, fnt); \
  NON_LDBL_TEST (fn, FIRST, vdouble2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, vdouble2, double, fnt); \
  NON_LDBL_TEST (fn, FIRST, Vdouble2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, Vdouble2, double, fnt); \
  NON_LDBL_TEST (fn, FIRST, vint2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, vint2, double, fnt); \
  NON_LDBL_TEST (fn, FIRST, Vint2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, Vint2, double, fnt); \
  NON_LDBL_TEST (fn, FIRST, vllong2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, vllong2, double, fnt); \
  NON_LDBL_TEST (fn, FIRST, Vllong2, double, fnt); \
  NON_LDBL_TEST (fn, SECOND, Vllong2, double, fnt);
#define BINARY_CTEST(fn, fnt) \
  BINARY_TEST (fn, fnt); \
  TEST (fn (vcfloat1, vfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, vfloat2), cfloat, fnt); \
  TEST (fn (vcfloat1, Vfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, Vfloat2), cfloat, fnt); \
  TEST (fn (vcldouble1, vldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, vldouble2), cldouble, fnt); \
  TEST (fn (vcldouble1, Vldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, Vldouble2), cldouble, fnt); \
  TEST (fn (vcfloat1, vfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, vfloat2), cfloat, fnt); \
  TEST (fn (vcfloat1, Vfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, Vfloat2), cfloat, fnt); \
  TEST (fn (vcldouble1, vldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, vldouble2), cldouble, fnt); \
  TEST (fn (vcldouble1, Vldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, Vldouble2), cldouble, fnt); \
  TEST (fn (vcfloat1, vcfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, vcfloat2), cfloat, fnt); \
  TEST (fn (vcfloat1, Vcfloat2), cfloat, fnt); \
  TEST (fn (Vcfloat1, Vcfloat2), cfloat, fnt); \
  TEST (fn (vcldouble1, vcldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, vcldouble2), cldouble, fnt); \
  TEST (fn (vcldouble1, Vcldouble2), cldouble, fnt); \
  TEST (fn (Vcldouble1, Vcldouble2), cldouble, fnt); \
  NON_LDBL_CTEST (fn, FIRST, vcldouble2, cldouble, fnt); \
  NON_LDBL_CTEST (fn, SECOND, vcldouble2, cldouble, fnt); \
  NON_LDBL_CTEST (fn, FIRST, Vcldouble2, cldouble, fnt); \
  NON_LDBL_CTEST (fn, SECOND, Vcldouble2, cldouble, fnt); \
  NON_LDBL_CTEST (fn, FIRST, vcdouble2, cdouble, fnt); \
  NON_LDBL_CTEST (fn, SECOND, vcdouble2, cdouble, fnt); \
  NON_LDBL_CTEST (fn, FIRST, Vcdouble2, cdouble, fnt); \
  NON_LDBL_CTEST (fn, SECOND, Vcdouble2, cdouble, fnt);

int
test_atan2 (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  BINARY_TEST (atan2, atan2);

  return result;
}

int
test_remquo (const int Vint4, const long long int Vllong4)
{
  int result = 0;
  int quo = 0;

#define my_remquo(x, y) remquo (x, y, &quo)
  BINARY_TEST (my_remquo, remquo);
#undef my_remquo

  return result;
}

int
test_pow (const int Vint4, const long long int Vllong4)
{
  int result = 0;

  BINARY_CTEST (pow, pow);

  return result;
}

/* Testing all arguments of fma would be just too expensive,
   so test just some.  */

int
test_fma_1 (const int Vint4, const long long int Vllong4)
{
  int result = 0;

#define my_fma(x, y) fma (x, y, vfloat3)
  BINARY_TEST (my_fma, fma);
#undef my_fma

  return result;
}

int
test_fma_2 (const int Vint4, const long long int Vllong4)
{
  int result = 0;

#define my_fma(x, y) fma (x, vfloat3, y)
  BINARY_TEST (my_fma, fma);
#undef my_fma

  return result;
}

int
test_fma_3 (const int Vint4, const long long int Vllong4)
{
  int result = 0;

#define my_fma(x, y) fma (Vfloat3, x, y)
  BINARY_TEST (my_fma, fma);
#undef my_fma

  return result;
}

int
test_fma_4 (const int Vint4, const long long int Vllong4)
{
  int result = 0;
  TEST (fma (vdouble1, Vdouble2, vllong3), double, fma);
  TEST (fma (vint1, Vint2, vint3), double, fma);
  TEST (fma (Vldouble1, vldouble2, Vldouble3), ldouble, fma);
  TEST (fma (vldouble1, vint2, Vdouble3), ldouble, fma);

  return result;
}

static int
do_test (void)
{
  int result;

  result = test_cos (vint1, vllong1);
  result |= test_fabs (vint1, vllong1);
  result |= test_conj (vint1, vllong1);
  result |= test_expm1 (vint1, vllong1);
  result |= test_lrint (vint1, vllong1);
  result |= test_ldexp (vint1, vllong1);
  result |= test_atan2 (vint1, vllong1);
  result |= test_remquo (vint1, vllong1);
  result |= test_pow (vint1, vllong1);
  result |= test_fma_1 (vint1, vllong1);
  result |= test_fma_2 (vint1, vllong1);
  result |= test_fma_3 (vint1, vllong1);
  result |= test_fma_4 (vint1, vllong1);

  return result;
}

/* Now generate the three functions.  */
#define HAVE_MAIN

#define F(name) name
#define TYPE double
#define CTYPE cdouble
#define T Tdouble
#define C Tcdouble
#include "test-tgmath2.c"

#define F(name) name##f
#define TYPE float
#define CTYPE cfloat
#define T Tfloat
#define C Tcfloat
#include "test-tgmath2.c"

#if LDBL_MANT_DIG > DBL_MANT_DIG
#define F(name) name##l
#define TYPE ldouble
#define CTYPE cldouble
#define T Tldouble
#define C Tcldouble
#include "test-tgmath2.c"
#endif

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

#else

#ifdef DEBUG
#define P() puts (__FUNCTION__); count++
#else
#define P() count++;
#endif

TYPE
(F(cos)) (TYPE x)
{
  counts[T][C_cos]++;
  P ();
  return x;
}

CTYPE
(F(ccos)) (CTYPE x)
{
  counts[C][C_cos]++;
  P ();
  return x;
}

TYPE
(F(fabs)) (TYPE x)
{
  counts[T][C_fabs]++;
  P ();
  return x;
}

TYPE
(F(cabs)) (CTYPE x)
{
  counts[T][C_cabs]++;
  P ();
  return x;
}

CTYPE
(F(conj)) (CTYPE x)
{
  counts[C][C_conj]++;
  P ();
  return x;
}

TYPE
(F(expm1)) (TYPE x)
{
  counts[T][C_expm1]++;
  P ();
  return x;
}

long int
(F(lrint)) (TYPE x)
{
  counts[T][C_lrint]++;
  P ();
  return x;
}

TYPE
(F(ldexp)) (TYPE x, int y)
{
  counts[T][C_ldexp]++;
  P ();
  return x + y;
}

TYPE
(F(atan2)) (TYPE x, TYPE y)
{
  counts[T][C_atan2]++;
  P ();
  return x + y;
}

TYPE
(F(remquo)) (TYPE x, TYPE y, int *z)
{
  counts[T][C_remquo]++;
  P ();
  return x + y + *z;
}

TYPE
(F(pow)) (TYPE x, TYPE y)
{
  counts[T][C_pow]++;
  P ();
  return x + y;
}

CTYPE
(F(cpow)) (CTYPE x, CTYPE y)
{
  counts[C][C_pow]++;
  P ();
  return x + y;
}

TYPE
(F(fma)) (TYPE x, TYPE y, TYPE z)
{
  counts[T][C_fma]++;
  P ();
  return x + y + z;
}

#undef F
#undef TYPE
#undef CTYPE
#undef T
#undef C
#undef P
#endif
