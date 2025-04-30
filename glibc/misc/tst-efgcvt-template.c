/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

/* This template provides testing for the *cvt family of functions,
   which deal with double or long double types.  In order to use the
   template, the following macros must be defined before inclusion of
   this template:

     FLOAT: The floating-point type, i.e. double or long double.

     ECVT: Appropriate *ecvt function for FLOAT, i.e. ecvt or qecvt.
     FCVT: Likewise for *fcvt, i.e. fcvt or qfcvt.
     ECVT_R: Likewise for *ecvt_r, i.e. ecvt_r or qecvt_r.
     FCVT_R: Likewise for *fcvt_r, i.e. fcvt_r or qfcvt_r.

     PRINTF_CONVERSION: The appropriate printf conversion specifier with
     length modifier for FLOAT, i.e. "%f" or "%Lf".

     EXTRA_ECVT_TESTS: Additional tests for the ecvt or qecvt function
     that are only relevant to a particular floating-point type and
     cannot be represented generically.  */

#ifndef _GNU_SOURCE
# define _GNU_SOURCE	1
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <support/check.h>

#define NAME(x) NAMEX(x)
#define NAMEX(x) #x

typedef struct
{
  FLOAT value;
  int ndigit;
  int decpt;
  char result[30];
} testcase;

typedef char * ((*efcvt_func) (FLOAT, int, int *, int *));

typedef int ((*efcvt_r_func) (FLOAT, int, int *, int *, char *, size_t));


static testcase ecvt_tests[] =
{
  { 0.0, 0, 1, "" },
  { 10.0, 0, 2, "" },
  { 10.0, 1, 2, "1" },
  { 10.0, 5, 2, "10000" },
  { -12.0, 5, 2, "12000" },
  { 0.2, 4, 0, "2000" },
  { 0.02, 4, -1, "2000" },
  { 5.5, 1, 1, "6" },
  { 1.0, -1, 1, "" },
  { 0.01, 2, -1, "10" },
  { 100.0, -2, 3, "" },
  { 100.0, -5, 3, "" },
  { 100.0, -4, 3, "" },
  { 100.01, -4, 3, "" },
  { 123.01, -4, 3, "" },
  { 126.71, -4, 3, "" },
  { 0.0, 4, 1, "0000" },
  EXTRA_ECVT_TESTS
  /* -1.0 is end marker.  */
  { -1.0, 0, 0, "" }
};

static testcase fcvt_tests[] =
{
  { 0.0, 0, 1, "0" },
  { 10.0, 0, 2, "10" },
  { 10.0, 1, 2, "100" },
  { 10.0, 4, 2, "100000" },
  { -12.0, 5, 2, "1200000" },
  { 0.2, 4, 0, "2000" },
  { 0.02, 4, -1, "200" },
  { 5.5, 1, 1, "55" },
  { 5.5, 0, 1, "6" },
  { 0.01, 2, -1, "1" },
  { 100.0, -2, 3, "100" },
  { 100.0, -5, 3, "100" },
  { 100.0, -4, 3, "100" },
  { 100.01, -4, 3, "100" },
  { 123.01, -4, 3, "100" },
  { 126.71, -4, 3, "100" },
  { 322.5, 16, 3, "3225000000000000000" },
  /* -1.0 is end marker.  */
  { -1.0, 0, 0, "" }
};

static void
output_error (const char *name, FLOAT value, int ndigit,
	      const char *exp_p, int exp_decpt, int exp_sign,
	      char *res_p, int res_decpt, int res_sign)
{
  printf ("%s returned wrong result for value: " PRINTF_CONVERSION
	  ", ndigits: %d\n",
	  name, value, ndigit);
  printf ("Result was p: \"%s\", decpt: %d, sign: %d\n",
	  res_p, res_decpt, res_sign);
  printf ("Should be  p: \"%s\", decpt: %d, sign: %d\n",
	  exp_p, exp_decpt, exp_sign);
  support_record_failure ();
}


static void
output_r_error (const char *name, FLOAT value, int ndigit,
		const char *exp_p, int exp_decpt, int exp_sign, int exp_return,
		char *res_p, int res_decpt, int res_sign, int res_return)
{
  printf ("%s returned wrong result for value: " PRINTF_CONVERSION
	  ", ndigits: %d\n",
	  name, value, ndigit);
  printf ("Result was buf: \"%s\", decpt: %d, sign: %d return value: %d\n",
	  res_p, res_decpt, res_sign, res_return);
  printf ("Should be  buf: \"%s\", decpt: %d, sign: %d\n",
	  exp_p, exp_decpt, exp_sign);
  support_record_failure ();
}

static void
test (testcase tests[], efcvt_func efcvt, const char *name)
{
  int no = 0;
  int decpt, sign;
  char *p;

  while (tests[no].value != -1.0)
    {
      p = efcvt (tests[no].value, tests[no].ndigit, &decpt, &sign);
      if (decpt != tests[no].decpt
	  || sign != (tests[no].value < 0)
	  || strcmp (p, tests[no].result) != 0)
	output_error (name, tests[no].value, tests[no].ndigit,
		      tests[no].result, tests[no].decpt,
		      (tests[no].value < 0),
		      p, decpt, sign);
      ++no;
    }
}

static void
test_r (testcase tests[], efcvt_r_func efcvt_r, const char *name)
{
  int no = 0;
  int decpt, sign, res;
  char buf [1024];


  while (tests[no].value != -1.0)
    {
      res = efcvt_r (tests[no].value, tests[no].ndigit, &decpt, &sign,
		     buf, sizeof (buf));
      if (res != 0
	  || decpt != tests[no].decpt
	  || sign != (tests[no].value < 0)
	  || strcmp (buf, tests[no].result) != 0)
	output_r_error (name, tests[no].value, tests[no].ndigit,
			tests[no].result, tests[no].decpt, 0,
			(tests[no].value < 0),
			buf, decpt, sign, res);
      ++no;
    }
}

static void
special (void)
{
  int decpt, sign, res;
  char *p;
  char buf [1024];

  p = ECVT (NAN, 10, &decpt, &sign);
  if (sign != 0 || strcmp (p, "nan") != 0)
    output_error (NAME (ECVT), NAN, 10, "nan", 0, 0, p, decpt, sign);

  p = ECVT (INFINITY, 10, &decpt, &sign);
  if (sign != 0 || strcmp (p, "inf") != 0)
    output_error (NAME (ECVT), INFINITY, 10, "inf", 0, 0, p, decpt, sign);

  /* Simply make sure these calls with large NDIGITs don't crash.  */
  (void) ECVT (123.456, 10000, &decpt, &sign);
  (void) FCVT (123.456, 10000, &decpt, &sign);

  /* Some tests for the reentrant functions.  */
  /* Use a too small buffer.  */
  res = ECVT_R (123.456, 10, &decpt, &sign, buf, 1);
  if (res == 0)
    {
      printf (NAME (ECVT_R) " with a too small buffer was succesful.\n");
      support_record_failure ();
    }
  res = FCVT_R (123.456, 10, &decpt, &sign, buf, 1);
  if (res == 0)
    {
      printf (NAME (FCVT_R) " with a too small buffer was succesful.\n");
      support_record_failure ();
    }
}


static int
do_test (void)
{
  test (ecvt_tests, ECVT, NAME (ECVT));
  test (fcvt_tests, FCVT, NAME (FCVT));
  test_r (ecvt_tests, ECVT_R, NAME (ECVT_R));
  test_r (fcvt_tests, FCVT_R, NAME (FCVT_R));
  special ();

  return 0;
}

#include <support/test-driver.c>
