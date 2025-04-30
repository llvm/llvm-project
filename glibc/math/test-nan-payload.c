/* Test nan functions payload handling (bug 16961).
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Avoid built-in functions.  */
#define WRAP_NAN(FUNC, STR) \
  ({ const char *volatile wns = (STR); FUNC (wns); })
#define WRAP_STRTO(FUNC, STR) \
  ({ const char *volatile wss = (STR); FUNC (wss, NULL); })

#define CHECK_IS_NAN(TYPE, A)			\
  do						\
    {						\
      if (isnan (A))				\
	puts ("PASS: " #TYPE " " #A);		\
      else					\
	{					\
	  puts ("FAIL: " #TYPE " " #A);		\
	  result = 1;				\
	}					\
    }						\
  while (0)

#define CHECK_SAME_NAN(TYPE, A, B)			\
  do							\
    {							\
      if (memcmp (&(A), &(B), sizeof (A)) == 0)		\
	puts ("PASS: " #TYPE " " #A " = " #B);		\
      else						\
	{						\
	  puts ("FAIL: " #TYPE " " #A " = " #B);	\
	  result = 1;					\
	}						\
    }							\
  while (0)

#define CHECK_DIFF_NAN(TYPE, A, B)			\
  do							\
    {							\
      if (memcmp (&(A), &(B), sizeof (A)) != 0)		\
	puts ("PASS: " #TYPE " " #A " != " #B);		\
      else						\
	{						\
	  puts ("FAIL: " #TYPE " " #A " != " #B);	\
	  result = 1;					\
	}						\
    }							\
  while (0)

/* Cannot test payloads by memcmp for formats where NaNs have padding
   bits.  */
#define CAN_TEST_EQ(MANT_DIG) ((MANT_DIG) != 64 && (MANT_DIG) != 106)

#define RUN_TESTS(TYPE, SFUNC, FUNC, MANT_DIG)		\
  do							\
    {							\
     TYPE n123 = WRAP_NAN (FUNC, "123");		\
     CHECK_IS_NAN (TYPE, n123);				\
     TYPE s123 = WRAP_STRTO (SFUNC, "NAN(123)");	\
     CHECK_IS_NAN (TYPE, s123);				\
     TYPE n456 = WRAP_NAN (FUNC, "456");		\
     CHECK_IS_NAN (TYPE, n456);				\
     TYPE s456 = WRAP_STRTO (SFUNC, "NAN(456)");	\
     CHECK_IS_NAN (TYPE, s456);				\
     TYPE n123x = WRAP_NAN (FUNC, "123)");		\
     CHECK_IS_NAN (TYPE, n123x);			\
     TYPE nemp = WRAP_NAN (FUNC, "");			\
     CHECK_IS_NAN (TYPE, nemp);				\
     TYPE semp = WRAP_STRTO (SFUNC, "NAN()");		\
     CHECK_IS_NAN (TYPE, semp);				\
     TYPE sx = WRAP_STRTO (SFUNC, "NAN");		\
     CHECK_IS_NAN (TYPE, sx);				\
     if (CAN_TEST_EQ (MANT_DIG))			\
       CHECK_SAME_NAN (TYPE, n123, s123);		\
     if (CAN_TEST_EQ (MANT_DIG))			\
       CHECK_SAME_NAN (TYPE, n456, s456);		\
     if (CAN_TEST_EQ (MANT_DIG))			\
       CHECK_SAME_NAN (TYPE, nemp, semp);		\
     if (CAN_TEST_EQ (MANT_DIG))			\
       CHECK_SAME_NAN (TYPE, n123x, sx);		\
     CHECK_DIFF_NAN (TYPE, n123, n456);			\
     CHECK_DIFF_NAN (TYPE, n123, nemp);			\
     CHECK_DIFF_NAN (TYPE, n123, n123x);		\
     CHECK_DIFF_NAN (TYPE, n456, nemp);			\
     CHECK_DIFF_NAN (TYPE, n456, n123x);		\
    }							\
  while (0)

static int
do_test (void)
{
  int result = 0;
  RUN_TESTS (float, strtof, nanf, FLT_MANT_DIG);
  RUN_TESTS (double, strtod, nan, DBL_MANT_DIG);
  RUN_TESTS (long double, strtold, nanl, LDBL_MANT_DIG);
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
