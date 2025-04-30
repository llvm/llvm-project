/* Support code for testing libm functions (compiled once per type).
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* Part of testsuite for libm.

   libm-test-support.c contains functions shared by tests of different
   libm functions and types; it is compiled once per type.
   libm-test-driver.c defines the main function, and various variables
   that are used to configure the code in libm-test-support.c for
   different types and for variants such as testing inline functions.

   The tests of individual functions are in .inc files processed by
   gen-libm-test.py, with the resulting files included together with
   libm-test-driver.c.

   The per-type headers included both before libm-test-support.c and
   for the tests of individual functions must define the following
   macros:

   FUNC(function): Convert general function name (like cos) to name
   with correct suffix (e.g. cosl or cosf).

   FLOAT: Floating-point type to test.

   BUILD_COMPLEX(real, imag): Create a complex number by calling a
   macro such as CMPLX.

   PREFIX: The prefix for <float.h> macros for the type (e.g. LDBL,
   DBL, or FLT).

   TYPE_STR: The name of the type as used in ulps files, as a string.

   ULP_IDX: The array indexes for ulps values for this function.

   LIT: Append the correct suffix to a literal.

   LITM: Append the correct suffix to an M_* macro name.

   FTOSTR: A function similar in type to strfromf which converts a
   FLOAT to a string.

   snan_value_MACRO: The macro such as SNAN for a signaling NaN for
   the type.

*/

/* Parameter handling is primitive in the moment:
   --verbose=[0..3] for different levels of output:
   0: only error count
   1: basic report on failed tests (default)
   2: full report on all tests
   -v for full output (equals --verbose=3)
   -u for generation of an ULPs file
 */

/* "Philosophy":

   This suite tests some aspects of the correct implementation of
   mathematical functions in libm.  Some simple, specific parameters
   are tested for correctness but there's no exhaustive
   testing.  Handling of specific inputs (e.g. infinity, not-a-number)
   is also tested.  Correct handling of exceptions is checked
   against.  These implemented tests should check all cases that are
   specified in ISO C99.

   NaN values: The payload of NaNs is set in inputs for functions
   where it is significant, and is examined in the outputs of some
   functions.

   Inline functions: Inlining functions should give an improvement in
   speed - but not in precission.  The inlined functions return
   reasonable values for a reasonable range of input values.  The
   result is not necessarily correct for all values and exceptions are
   not correctly raised in all cases.  Problematic input and return
   values are infinity, not-a-number and minus zero.  This suite
   therefore does not check these specific inputs and the exception
   handling for inlined mathematical functions - just the "reasonable"
   values are checked.

   Beware: The tests might fail for any of the following reasons:
   - Tests are wrong
   - Functions are wrong
   - Floating Point Unit not working properly
   - Compiler has errors

   With e.g. gcc 2.7.2.2 the test for cexp fails because of a compiler error.


   To Do: All parameter should be numbers that can be represented as
   exact floating point values.  Currently some values cannot be
   represented exactly and therefore the result is not the expected
   result.  For this we will use 36 digits so that numbers can be
   represented exactly.  */

#include "libm-test-support.h"

#include <argp.h>
#include <errno.h>
#include <string.h>

/* This header defines func_ulps, func_real_ulps and func_imag_ulps
   arrays.  */
#include "libm-test-ulps.h"

/* Maximum character buffer to store a stringitized FLOAT value.  */
#define FSTR_MAX (128)

#define ulps_file_name "ULPs"	/* Name of the ULPs file.  */
static FILE *ulps_file;		/* File to document difference.  */
static int output_ulps;		/* Should ulps printed?  */
static char *output_dir;	/* Directory where generated files will be written.  */

static int noErrors;	/* number of errors */
static int noTests;	/* number of tests (without testing exceptions) */
static int noExcTests;	/* number of tests for exception flags */
static int noErrnoTests;/* number of tests for errno values */

static int verbose;
static int output_max_error;	/* Should the maximal errors printed?  */
static int output_points;	/* Should the single function results printed?  */
static int ignore_max_ulp;	/* Should we ignore max_ulp?  */
static int test_ibm128;		/* Is argument or result IBM long double?  */

static FLOAT max_error, real_max_error, imag_max_error;

static FLOAT prev_max_error, prev_real_max_error, prev_imag_max_error;

static FLOAT max_valid_error;

/* Sufficient numbers of digits to represent any floating-point value
   unambiguously (for any choice of the number of bits in the first
   hex digit, in the case of TYPE_HEX_DIG).  When used with printf
   formats where the precision counts only digits after the point, 1
   is subtracted from these values. */
#define TYPE_DECIMAL_DIG __CONCATX (PREFIX, _DECIMAL_DIG)
#define TYPE_HEX_DIG ((MANT_DIG + 6) / 4)

/* Converts VALUE (a floating-point number) to string and writes it to DEST.
   PRECISION specifies the number of fractional digits that should be printed.
   CONVERSION is the conversion specifier, such as in printf, e.g. 'f' or 'a'.
   The output is prepended with an empty space if VALUE is non-negative.  */
static void
fmt_ftostr (char *dest, size_t size, int precision, const char *conversion,
	    FLOAT value)
{
  char format[64];
  char *ptr_format;
  int ret;

  /* Generate the format string.  */
  ptr_format = stpcpy (format, "%.");
  ret = sprintf (ptr_format, "%d", precision);
  ptr_format += ret;
  ptr_format = stpcpy (ptr_format, conversion);

  /* Add a space to the beginning of the output string, if the floating-point
     number is non-negative.  This mimics the behavior of the space (' ') flag
     in snprintf, which is not available on strfrom.  */
  if (! signbit (value))
    {
      *dest = ' ';
      dest++;
      size--;
    }

  /* Call the float to string conversion function, e.g.: strfromd.  */
  FTOSTR (dest, size, format, value);
}

/* Compare KEY (a string, with the name of a function) with ULP (a
   pointer to a struct ulp_data structure), returning a value less
   than, equal to or greater than zero for use in bsearch.  */

static int
compare_ulp_data (const void *key, const void *ulp)
{
  const char *keystr = key;
  const struct ulp_data *ulpdat = ulp;
  return strcmp (keystr, ulpdat->name);
}

static const int ulp_idx = ULP_IDX;

/* Return the ulps for NAME in array DATA with NMEMB elements, or 0 if
   no ulps listed.  */

static FLOAT
find_ulps (const char *name, const struct ulp_data *data, size_t nmemb)
{
  const struct ulp_data *entry = bsearch (name, data, nmemb, sizeof (*data),
					  compare_ulp_data);
  if (entry == NULL)
    return 0;
  else
    return entry->max_ulp[ulp_idx];
}

void
init_max_error (const char *name, int exact, int testing_ibm128)
{
  max_error = 0;
  real_max_error = 0;
  imag_max_error = 0;
  test_ibm128 = testing_ibm128;
  prev_max_error = find_ulps (name, func_ulps,
			      sizeof (func_ulps) / sizeof (func_ulps[0]));
  prev_real_max_error = find_ulps (name, func_real_ulps,
				   (sizeof (func_real_ulps)
				    / sizeof (func_real_ulps[0])));
  prev_imag_max_error = find_ulps (name, func_imag_ulps,
				   (sizeof (func_imag_ulps)
				    / sizeof (func_imag_ulps[0])));
  if (testing_ibm128)
    /* The documented accuracy of IBM long double division is 3ulp
       (see libgcc/config/rs6000/ibm-ldouble-format), so do not
       require better accuracy for libm functions that are exactly
       defined for other formats.  */
    max_valid_error = exact ? 3 : 16;
  else
    max_valid_error = exact ? 0 : 9;
  prev_max_error = (prev_max_error <= max_valid_error
		    ? prev_max_error
		    : max_valid_error);
  prev_real_max_error = (prev_real_max_error <= max_valid_error
			 ? prev_real_max_error
			 : max_valid_error);
  prev_imag_max_error = (prev_imag_max_error <= max_valid_error
			 ? prev_imag_max_error
			 : max_valid_error);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}

static void
set_max_error (FLOAT current, FLOAT *curr_max_error)
{
  if (current > *curr_max_error && current <= max_valid_error)
    *curr_max_error = current;
}


/* Print a FLOAT.  */
static void
print_float (FLOAT f)
{
  /* As printf doesn't differ between a sNaN and a qNaN, do this manually.  */
  if (issignaling (f))
    printf ("sNaN\n");
  else if (isnan (f))
    printf ("qNaN\n");
  else
    {
      char fstrn[FSTR_MAX], fstrx[FSTR_MAX];
      fmt_ftostr (fstrn, FSTR_MAX, TYPE_DECIMAL_DIG - 1, "e", f);
      fmt_ftostr (fstrx, FSTR_MAX, TYPE_HEX_DIG - 1, "a", f);
      printf ("%s  %s\n", fstrn, fstrx);
    }
}

/* Should the message print to screen?  This depends on the verbose flag,
   and the test status.  */
static int
print_screen (int ok)
{
  if (output_points
      && (verbose > 1
	  || (verbose == 1 && ok == 0)))
    return 1;
  return 0;
}


/* Should the message print to screen?  This depends on the verbose flag,
   and the test status.  */
static int
print_screen_max_error (int ok)
{
  if (output_max_error
      && (verbose > 1
	  || ((verbose == 1) && (ok == 0))))
    return 1;
  return 0;
}

/* Update statistic counters.  */
static void
update_stats (int ok)
{
  ++noTests;
  if (!ok)
    ++noErrors;
}

static void
print_function_ulps (const char *function_name, FLOAT ulp)
{
  if (output_ulps)
    {
      char ustrn[FSTR_MAX];
      FTOSTR (ustrn, FSTR_MAX, "%.0f", FUNC (ceil) (ulp));
      fprintf (ulps_file, "Function: \"%s\":\n", function_name);
      fprintf (ulps_file, "%s: %s\n", qtype_str, ustrn);
    }
}


static void
print_complex_function_ulps (const char *function_name, FLOAT real_ulp,
			     FLOAT imag_ulp)
{
  if (output_ulps)
    {
      char fstrn[FSTR_MAX];
      if (real_ulp != 0.0)
	{
	  FTOSTR (fstrn, FSTR_MAX, "%.0f", FUNC (ceil) (real_ulp));
	  fprintf (ulps_file, "Function: Real part of \"%s\":\n", function_name);
	  fprintf (ulps_file, "%s: %s\n", qtype_str, fstrn);
	}
      if (imag_ulp != 0.0)
	{
	  FTOSTR (fstrn, FSTR_MAX, "%.0f", FUNC (ceil) (imag_ulp));
	  fprintf (ulps_file, "Function: Imaginary part of \"%s\":\n", function_name);
	  fprintf (ulps_file, "%s: %s\n", qtype_str, fstrn);
	}


    }
}



/* Test if Floating-Point stack hasn't changed */
static void
fpstack_test (const char *test_name)
{
#if defined (__i386__) || defined (__x86_64__)
  static int old_stack;
  int sw;

  asm ("fnstsw" : "=a" (sw));
  sw >>= 11;
  sw &= 7;

  if (sw != old_stack)
    {
      printf ("FP-Stack wrong after test %s (%d, should be %d)\n",
	      test_name, sw, old_stack);
      ++noErrors;
      old_stack = sw;
    }
#endif
}


void
print_max_error (const char *func_name)
{
  int ok = 0;

  if (max_error == 0.0 || (max_error <= prev_max_error && !ignore_max_ulp))
    {
      ok = 1;
    }

  if (!ok)
    print_function_ulps (func_name, max_error);


  if (print_screen_max_error (ok))
    {
      char mestr[FSTR_MAX], pmestr[FSTR_MAX];
      FTOSTR (mestr, FSTR_MAX, "%.0f", FUNC (ceil) (max_error));
      FTOSTR (pmestr, FSTR_MAX, "%.0f", FUNC (ceil) (prev_max_error));
      printf ("Maximal error of `%s'\n", func_name);
      printf (" is      : %s ulp\n", mestr);
      printf (" accepted: %s ulp\n", pmestr);
    }

  update_stats (ok);
}


void
print_complex_max_error (const char *func_name)
{
  int real_ok = 0, imag_ok = 0, ok;

  if (real_max_error == 0
      || (real_max_error <= prev_real_max_error && !ignore_max_ulp))
    {
      real_ok = 1;
    }

  if (imag_max_error == 0
      || (imag_max_error <= prev_imag_max_error && !ignore_max_ulp))
    {
      imag_ok = 1;
    }

  ok = real_ok && imag_ok;

  if (!ok)
    print_complex_function_ulps (func_name,
				 real_ok ? 0 : real_max_error,
				 imag_ok ? 0 : imag_max_error);

  if (print_screen_max_error (ok))
    {
      char rmestr[FSTR_MAX], prmestr[FSTR_MAX];
      char imestr[FSTR_MAX], pimestr[FSTR_MAX];
      FTOSTR (rmestr, FSTR_MAX, "%.0f", FUNC (ceil) (real_max_error));
      FTOSTR (prmestr, FSTR_MAX, "%.0f", FUNC (ceil) (prev_real_max_error));
      FTOSTR (imestr, FSTR_MAX, "%.0f", FUNC (ceil) (imag_max_error));
      FTOSTR (pimestr, FSTR_MAX, "%.0f", FUNC (ceil) (prev_imag_max_error));
      printf ("Maximal error of real part of: %s\n", func_name);
      printf (" is      : %s ulp\n", rmestr);
      printf (" accepted: %s ulp\n", prmestr);
      printf ("Maximal error of imaginary part of: %s\n", func_name);
      printf (" is      : %s ulp\n", imestr);
      printf (" accepted: %s ulp\n", pimestr);
    }

  update_stats (ok);
}


#if FE_ALL_EXCEPT
/* Test whether a given exception was raised.  */
static void
test_single_exception (const char *test_name,
		       int exception,
		       int exc_flag,
		       int fe_flag,
		       const char *flag_name)
{
  int ok = 1;
  if (exception & exc_flag)
    {
      if (fetestexcept (fe_flag))
	{
	  if (print_screen (1))
	    printf ("Pass: %s: Exception \"%s\" set\n", test_name, flag_name);
	}
      else
	{
	  ok = 0;
	  if (print_screen (0))
	    printf ("Failure: %s: Exception \"%s\" not set\n",
		    test_name, flag_name);
	}
    }
  else
    {
      if (fetestexcept (fe_flag))
	{
	  ok = 0;
	  if (print_screen (0))
	    printf ("Failure: %s: Exception \"%s\" set\n",
		    test_name, flag_name);
	}
      else
	{
	  if (print_screen (1))
	    printf ("%s: Exception \"%s\" not set\n", test_name,
		    flag_name);
	}
    }
  if (!ok)
    ++noErrors;
}
#endif

/* Test whether exceptions given by EXCEPTION are raised.  Ignore thereby
   allowed but not required exceptions.
*/
static void
test_exceptions (const char *test_name, int exception)
{
  if (flag_test_exceptions && EXCEPTION_TESTS (FLOAT))
    {
      ++noExcTests;
#ifdef FE_DIVBYZERO
      if ((exception & DIVIDE_BY_ZERO_EXCEPTION_OK) == 0)
	test_single_exception (test_name, exception,
			       DIVIDE_BY_ZERO_EXCEPTION, FE_DIVBYZERO,
			       "Divide by zero");
#endif
#ifdef FE_INVALID
      if ((exception & INVALID_EXCEPTION_OK) == 0)
	test_single_exception (test_name, exception,
			       INVALID_EXCEPTION, FE_INVALID,
			       "Invalid operation");
#endif
#ifdef FE_OVERFLOW
      if ((exception & OVERFLOW_EXCEPTION_OK) == 0)
	test_single_exception (test_name, exception, OVERFLOW_EXCEPTION,
			       FE_OVERFLOW, "Overflow");
#endif
      /* Spurious "underflow" and "inexact" exceptions are always
	 allowed for IBM long double, in line with the underlying
	 arithmetic.  */
#ifdef FE_UNDERFLOW
      if ((exception & UNDERFLOW_EXCEPTION_OK) == 0
	  && !(test_ibm128
	       && (exception & UNDERFLOW_EXCEPTION) == 0))
	test_single_exception (test_name, exception, UNDERFLOW_EXCEPTION,
			       FE_UNDERFLOW, "Underflow");
#endif
#ifdef FE_INEXACT
      if ((exception & (INEXACT_EXCEPTION | NO_INEXACT_EXCEPTION)) != 0
	  && !(test_ibm128
	       && (exception & NO_INEXACT_EXCEPTION) != 0))
	test_single_exception (test_name, exception, INEXACT_EXCEPTION,
			       FE_INEXACT, "Inexact");
#endif
    }
  feclearexcept (FE_ALL_EXCEPT);
}

/* Test whether errno for TEST_NAME, set to ERRNO_VALUE, has value
   EXPECTED_VALUE (description EXPECTED_NAME).  */
static void
test_single_errno (const char *test_name, int errno_value,
		   int expected_value, const char *expected_name)
{
  if (errno_value == expected_value)
    {
      if (print_screen (1))
	printf ("Pass: %s: errno set to %d (%s)\n", test_name, errno_value,
		expected_name);
    }
  else
    {
      ++noErrors;
      if (print_screen (0))
	printf ("Failure: %s: errno set to %d, expected %d (%s)\n",
		test_name, errno_value, expected_value, expected_name);
    }
}

/* Test whether errno (value ERRNO_VALUE) has been for TEST_NAME set
   as required by EXCEPTIONS.  */
static void
test_errno (const char *test_name, int errno_value, int exceptions)
{
  if (flag_test_errno)
    {
      ++noErrnoTests;
      if (exceptions & ERRNO_UNCHANGED)
	test_single_errno (test_name, errno_value, 0, "unchanged");
      if (exceptions & ERRNO_EDOM)
	test_single_errno (test_name, errno_value, EDOM, "EDOM");
      if (exceptions & ERRNO_ERANGE)
	test_single_errno (test_name, errno_value, ERANGE, "ERANGE");
    }
}

/* Returns the number of ulps that GIVEN is away from EXPECTED.  */
#define ULPDIFF(given, expected) \
	(FUNC(fabs) ((given) - (expected)) / ulp (expected))

/* Returns the size of an ulp for VALUE.  */
static FLOAT
ulp (FLOAT value)
{
  FLOAT ulp;

  switch (fpclassify (value))
    {
      case FP_ZERO:
	/* We compute the distance to the next FP which is the same as the
	   value of the smallest subnormal number. Previously we used
	   2^-(MANT_DIG - 1) which is too large a value to be useful. Note that we
	   can't use ilogb(0), since that isn't a valid thing to do. As a point
	   of comparison Java's ulp returns the next normal value e.g.
	   2^(1 - MAX_EXP) for ulp(0), but that is not what we want for
	   glibc.  */
	/* Fall through...  */
      case FP_SUBNORMAL:
        /* The next closest subnormal value is a constant distance away.  */
	ulp = FUNC(ldexp) (1.0, MIN_EXP - MANT_DIG);
	break;

      case FP_NORMAL:
	ulp = FUNC(ldexp) (1.0, FUNC(ilogb) (value) - MANT_DIG + 1);
	break;

      default:
	/* It should never happen. */
	abort ();
	break;
    }
  return ulp;
}

static void
check_float_internal (const char *test_name, FLOAT computed, FLOAT expected,
		      int exceptions,
		      FLOAT *curr_max_error, FLOAT max_ulp)
{
  int ok = 0;
  int print_diff = 0;
  FLOAT diff = 0;
  FLOAT ulps = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  if (issignaling (computed) && issignaling (expected))
    {
      if ((exceptions & TEST_NAN_SIGN) != 0
	  && signbit (computed) != signbit (expected))
	{
	  ok = 0;
	  printf ("signaling NaN has wrong sign.\n");
	}
      else if ((exceptions & TEST_NAN_PAYLOAD) != 0
	       && (FUNC (getpayload) (&computed)
		   != FUNC (getpayload) (&expected)))
	{
	  ok = 0;
	  printf ("signaling NaN has wrong payload.\n");
	}
      else
	ok = 1;
    }
  else if (issignaling (computed) || issignaling (expected))
    ok = 0;
  else if (isnan (computed) && isnan (expected))
    {
      if ((exceptions & TEST_NAN_SIGN) != 0
	  && signbit (computed) != signbit (expected))
	{
	  ok = 0;
	  printf ("quiet NaN has wrong sign.\n");
	}
      else if ((exceptions & TEST_NAN_PAYLOAD) != 0
	       && (FUNC (getpayload) (&computed)
		   != FUNC (getpayload) (&expected)))
	{
	  ok = 0;
	  printf ("quiet NaN has wrong payload.\n");
	}
      else
	ok = 1;
    }
  else if (isinf (computed) && isinf (expected))
    {
      /* Test for sign of infinities.  */
      if ((exceptions & IGNORE_ZERO_INF_SIGN) == 0
	  && signbit (computed) != signbit (expected))
	{
	  ok = 0;
	  printf ("infinity has wrong sign.\n");
	}
      else
	ok = 1;
    }
  /* Don't calculate ULPs for infinities or any kind of NaNs.  */
  else if (isinf (computed) || isnan (computed)
	   || isinf (expected) || isnan (expected))
    ok = 0;
  else
    {
      diff = FUNC(fabs) (computed - expected);
      ulps = ULPDIFF (computed, expected);
      set_max_error (ulps, curr_max_error);
      print_diff = 1;
      if ((exceptions & IGNORE_ZERO_INF_SIGN) == 0
	  && computed == 0.0 && expected == 0.0
	  && signbit(computed) != signbit (expected))
	ok = 0;
      else if (ulps <= max_ulp && !ignore_max_ulp)
	ok = 1;
      else
	ok = 0;
    }
  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure: ");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         ");
      print_float (computed);
      printf (" should be:  ");
      print_float (expected);
      if (print_diff)
	{
	  char dstrn[FSTR_MAX], dstrx[FSTR_MAX];
	  char ustrn[FSTR_MAX], mustrn[FSTR_MAX];
	  fmt_ftostr (dstrn, FSTR_MAX, TYPE_DECIMAL_DIG - 1, "e", diff);
	  fmt_ftostr (dstrx, FSTR_MAX, TYPE_HEX_DIG - 1, "a", diff);
	  fmt_ftostr (ustrn, FSTR_MAX, 4, "f", ulps);
	  fmt_ftostr (mustrn, FSTR_MAX, 4, "f", max_ulp);
	  printf (" difference: %s  %s\n", dstrn, dstrx);
	  printf (" ulp       : %s\n", ustrn);
	  printf (" max.ulp   : %s\n", mustrn);
	}
    }
  update_stats (ok);

 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


void
check_float (const char *test_name, FLOAT computed, FLOAT expected,
	     int exceptions)
{
  check_float_internal (test_name, computed, expected,
			exceptions, &max_error, prev_max_error);
}


void
check_complex (const char *test_name, CFLOAT computed,
	       CFLOAT expected,
	       int exception)
{
  FLOAT part_comp, part_exp;
  char *str;

  if (asprintf (&str, "Real part of: %s", test_name) == -1)
    abort ();

  part_comp = __real__ computed;
  part_exp = __real__ expected;

  check_float_internal (str, part_comp, part_exp,
			exception, &real_max_error, prev_real_max_error);
  free (str);

  if (asprintf (&str, "Imaginary part of: %s", test_name) == -1)
    abort ();

  part_comp = __imag__ computed;
  part_exp = __imag__ expected;

  /* Don't check again for exceptions or errno, just pass through the
     other relevant flags.  */
  check_float_internal (str, part_comp, part_exp,
			exception & (IGNORE_ZERO_INF_SIGN
				     | TEST_NAN_SIGN
				     | IGNORE_RESULT),
			&imag_max_error, prev_imag_max_error);
  free (str);
}


/* Check that computed and expected values are equal (int values).  */
void
check_int (const char *test_name, int computed, int expected,
	   int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if (computed == expected)
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure: ");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %d\n", computed);
      printf (" should be:  %d\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


/* Check that computed and expected values are equal (long int values).  */
void
check_long (const char *test_name, long int computed, long int expected,
	    int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if (computed == expected)
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure: ");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %ld\n", computed);
      printf (" should be:  %ld\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


/* Check that computed value is true/false.  */
void
check_bool (const char *test_name, int computed, int expected,
	    int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if ((computed == 0) == (expected == 0))
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure: ");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %d\n", computed);
      printf (" should be:  %d\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


/* check that computed and expected values are equal (long int values) */
void
check_longlong (const char *test_name, long long int computed,
		long long int expected,
		int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if (computed == expected)
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure:");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %lld\n", computed);
      printf (" should be:  %lld\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


/* Check that computed and expected values are equal (intmax_t values).  */
void
check_intmax_t (const char *test_name, intmax_t computed,
		intmax_t expected, int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if (computed == expected)
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure:");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %jd\n", computed);
      printf (" should be:  %jd\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}


/* Check that computed and expected values are equal (uintmax_t values).  */
void
check_uintmax_t (const char *test_name, uintmax_t computed,
		 uintmax_t expected, int exceptions)
{
  int ok = 0;
  int errno_value = errno;

  test_exceptions (test_name, exceptions);
  test_errno (test_name, errno_value, exceptions);
  if (exceptions & IGNORE_RESULT)
    goto out;
  noTests++;
  if (computed == expected)
    ok = 1;

  if (print_screen (ok))
    {
      if (!ok)
	printf ("Failure:");
      printf ("Test: %s\n", test_name);
      printf ("Result:\n");
      printf (" is:         %ju\n", computed);
      printf (" should be:  %ju\n", expected);
    }

  update_stats (ok);
 out:
  fpstack_test (test_name);
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
}

/* Return whether a test with flags EXCEPTIONS should be run.  */
int
enable_test (int exceptions)
{
  if (exceptions & XFAIL_TEST)
    return 0;
  if ((!SNAN_TESTS (FLOAT) || !snan_tests_arg)
      && (exceptions & TEST_SNAN) != 0)
    return 0;
  if (flag_test_mathvec && (exceptions & NO_TEST_MATHVEC) != 0)
    return 0;

  return 1;
}

static void
initialize (void)
{
  fpstack_test ("start *init*");

  /* Clear all exceptions.  From now on we must not get random exceptions.  */
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;

  /* Test to make sure we start correctly.  */
  fpstack_test ("end *init*");
}

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { "verbose", 'v', "NUMBER", 0, "Level of verbosity (0..3)"},
  { "ulps-file", 'u', NULL, 0, "Output ulps to file ULPs"},
  { "no-max-error", 'f', NULL, 0,
    "Don't output maximal errors of functions"},
  { "no-points", 'p', NULL, 0,
    "Don't output results of functions invocations"},
  { "ignore-max-ulp", 'i', "yes/no", 0,
    "Ignore given maximal errors"},
  { "output-dir", 'o', "DIR", 0,
    "Directory where generated files will be placed"},
  { NULL, 0, NULL, 0, NULL }
};

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, NULL, doc,
};


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'f':
      output_max_error = 0;
      break;
    case 'i':
      if (strcmp (arg, "yes") == 0)
	ignore_max_ulp = 1;
      else if (strcmp (arg, "no") == 0)
	ignore_max_ulp = 0;
      break;
    case 'o':
      output_dir = (char *) malloc (strlen (arg) + 1);
      if (output_dir != NULL)
	strcpy (output_dir, arg);
      else
        return errno;
      break;
    case 'p':
      output_points = 0;
      break;
    case 'u':
      output_ulps = 1;
      break;
    case 'v':
      if (optarg)
	verbose = (unsigned int) strtoul (optarg, NULL, 0);
      else
	verbose = 3;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Verify that our ulp () implementation is behaving as expected
   or abort.  */
static void
check_ulp (void)
{
   FLOAT ulps, ulpx, value;
   int i;
   /* Check ulp of zero is a subnormal value...  */
   ulps = ulp (0x0.0p0);
   if (fpclassify (ulps) != FP_SUBNORMAL)
     {
       fprintf (stderr, "ulp (0x0.0p0) is not FP_SUBNORMAL!\n");
       exit (EXIT_FAILURE);
     }
   /* Check that the ulp of one is a normal value... */
   ulps = ulp (LIT(1.0));
   if (fpclassify (ulps) != FP_NORMAL)
     {
       fprintf (stderr, "ulp (1.0L) is not FP_NORMAL\n");
       exit (EXIT_FAILURE);
     }

   /* Compute the next subnormal value using nextafter to validate ulp.
      We allow +/- 1 ulp around the represented value.  */
   value = FUNC(nextafter) (0, 1);
   ulps = ULPDIFF (value, 0);
   ulpx = ulp (LIT(1.0));
   if (ulps < (LIT(1.0) - ulpx) || ulps > (LIT(1.0) + ulpx))
     {
       fprintf (stderr, "Value outside of 1 +/- 1ulp.\n");
       exit (EXIT_FAILURE);
     }
   /* Compute the nearest representable number from 10 towards 20.
      The result is 10 + 1ulp.  We use this to check the ulp function.
      We allow +/- 1 ulp around the represented value.  */
   value = FUNC(nextafter) (10, 20);
   ulps = ULPDIFF (value, 10);
   ulpx = ulp (LIT(1.0));
   if (ulps < (LIT(1.0) - ulpx) || ulps > (LIT(1.0) + ulpx))
     {
       fprintf (stderr, "Value outside of 1 +/- 1ulp.\n");
       exit (EXIT_FAILURE);
     }
   /* This gives one more ulp.  */
   value = FUNC(nextafter) (value, 20);
   ulps = ULPDIFF (value, 10);
   ulpx = ulp (LIT(2.0));
   if (ulps < (LIT(2.0) - ulpx) || ulps > (LIT(2.0) + ulpx))
     {
       fprintf (stderr, "Value outside of 2 +/- 1ulp.\n");
       exit (EXIT_FAILURE);
     }
   /* And now calculate 100 ulp.  */
   for (i = 2; i < 100; i++)
     value = FUNC(nextafter) (value, 20);
   ulps = ULPDIFF (value, 10);
   ulpx = ulp (LIT(100.0));
   if (ulps < (LIT(100.0) - ulpx) || ulps > (LIT(100.0) + ulpx))
     {
       fprintf (stderr, "Value outside of 100 +/- 1ulp.\n");
       exit (EXIT_FAILURE);
     }
}

/* Do all initialization for a test run with arguments given by ARGC
   and ARGV.  */
void
libm_test_init (int argc, char **argv)
{
  int remaining;
  char *ulps_file_path;
  size_t dir_len = 0;

  verbose = 1;
  output_ulps = 0;
  output_max_error = 1;
  output_points = 1;
  output_dir = NULL;
  /* XXX set to 0 for releases.  */
  ignore_max_ulp = 0;

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (remaining != argc)
    {
      fprintf (stderr, "wrong number of arguments");
      argp_help (&argp, stdout, ARGP_HELP_SEE, program_invocation_short_name);
      exit (EXIT_FAILURE);
    }

  if (output_ulps)
    {
      if (output_dir != NULL)
	dir_len = strlen (output_dir);
      ulps_file_path = (char *) malloc (dir_len + strlen (ulps_file_name) + 1);
      if (ulps_file_path == NULL)
        {
	  perror ("can't allocate path for `ULPs' file: ");
	  exit (1);
        }
      sprintf (ulps_file_path, "%s%s", output_dir == NULL ? "" : output_dir, ulps_file_name);
      ulps_file = fopen (ulps_file_path, "a");
      if (ulps_file == NULL)
	{
	  perror ("can't open file `ULPs' for writing: ");
	  exit (1);
	}
    }


  initialize ();
  fputs (test_msg, stdout);

  check_ulp ();
}

/* Process the test results, returning the exit status.  */
int
libm_test_finish (void)
{
  if (output_ulps)
    fclose (ulps_file);

  printf ("\nTest suite completed:\n");
  printf ("  %d test cases plus %d tests for exception flags and\n"
	  "    %d tests for errno executed.\n",
	  noTests, noExcTests, noErrnoTests);
  if (noErrors)
    {
      printf ("  %d errors occurred.\n", noErrors);
      return 1;
    }
  printf ("  All tests passed successfully.\n");

  return 0;
}
