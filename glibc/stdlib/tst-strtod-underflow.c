/* Test for strtod handling of arguments that may cause floating-point
   underflow.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <tininess.h>

enum underflow_case
  {
    /* Result is exact or outside the subnormal range.  */
    UNDERFLOW_NONE,
    /* Result has magnitude at most half way between the largest
       subnormal value and the smallest positive normal value, and is
       not exact, so underflows in all rounding modes and independent
       of how tininess is detected.  */
    UNDERFLOW_ALWAYS,
    /* Result is positive, with magnitude larger than half way between
       the largest subnormal value and the least positive normal
       value, but would underflow when rounded to nearest to normal
       precision, so underflows after rounding in all modes except
       rounding upward.  */
    UNDERFLOW_EXCEPT_UPWARD,
    /* Likewise, for a negative result, underflowing after rounding
       except when rounding downward.  */
    UNDERFLOW_EXCEPT_DOWNWARD,
    /* Result is positive, with magnitude at least three quarters of
       the way from the largest subnormal value to the smallest
       positive normal value, so underflows after rounding only when
       rounding downward or toward zero.  */
    UNDERFLOW_ONLY_DOWNWARD_ZERO,
    /* Likewise, for a negative result, underflowing after rounding
       only when rounding upward or toward zero.  */
    UNDERFLOW_ONLY_UPWARD_ZERO,
  };

struct test
{
  const char *s;
  enum underflow_case c;
};

static const struct test tests[] =
  {
    { "0x1p-1022", UNDERFLOW_NONE },
    { "-0x1p-1022", UNDERFLOW_NONE },
    { "0x0p-10000000000000000000000000", UNDERFLOW_NONE },
    { "-0x0p-10000000000000000000000000", UNDERFLOW_NONE },
    { "0x1p-10000000000000000000000000", UNDERFLOW_ALWAYS },
    { "-0x1p-10000000000000000000000000", UNDERFLOW_ALWAYS },
    { "0x1.000000000000000000001p-1022", UNDERFLOW_NONE },
    { "-0x1.000000000000000000001p-1022", UNDERFLOW_NONE },
    { "0x1p-1075", UNDERFLOW_ALWAYS },
    { "-0x1p-1075", UNDERFLOW_ALWAYS },
    { "0x1p-1023", UNDERFLOW_NONE },
    { "-0x1p-1023", UNDERFLOW_NONE },
    { "0x1p-1074", UNDERFLOW_NONE },
    { "-0x1p-1074", UNDERFLOW_NONE },
    { "0x1.ffffffffffffep-1023", UNDERFLOW_NONE },
    { "-0x1.ffffffffffffep-1023", UNDERFLOW_NONE },
    { "0x1.fffffffffffffp-1023", UNDERFLOW_ALWAYS },
    { "-0x1.fffffffffffffp-1023", UNDERFLOW_ALWAYS },
    { "0x1.fffffffffffff0001p-1023", UNDERFLOW_EXCEPT_UPWARD },
    { "-0x1.fffffffffffff0001p-1023", UNDERFLOW_EXCEPT_DOWNWARD },
    { "0x1.fffffffffffff7fffp-1023", UNDERFLOW_EXCEPT_UPWARD },
    { "-0x1.fffffffffffff7fffp-1023", UNDERFLOW_EXCEPT_DOWNWARD },
    { "0x1.fffffffffffff8p-1023", UNDERFLOW_ONLY_DOWNWARD_ZERO },
    { "-0x1.fffffffffffff8p-1023", UNDERFLOW_ONLY_UPWARD_ZERO },
    { "0x1.fffffffffffffffffp-1023", UNDERFLOW_ONLY_DOWNWARD_ZERO },
    { "-0x1.fffffffffffffffffp-1023", UNDERFLOW_ONLY_UPWARD_ZERO },
  };

/* Return whether to expect underflow from a particular testcase, in a
   given rounding mode.  */

static bool
expect_underflow (enum underflow_case c, int rm)
{
  if (c == UNDERFLOW_NONE)
    return false;
  if (c == UNDERFLOW_ALWAYS)
    return true;
  if (TININESS_AFTER_ROUNDING)
    {
      switch (rm)
	{
#ifdef FE_DOWNWARD
	case FE_DOWNWARD:
	  return (c == UNDERFLOW_EXCEPT_UPWARD
		  || c == UNDERFLOW_ONLY_DOWNWARD_ZERO);
#endif

#ifdef FE_TOWARDZERO
	case FE_TOWARDZERO:
	  return true;
#endif

#ifdef FE_UPWARD
	case FE_UPWARD:
	  return (c == UNDERFLOW_EXCEPT_DOWNWARD
		  || c == UNDERFLOW_ONLY_UPWARD_ZERO);
#endif

	default:
	  return (c == UNDERFLOW_EXCEPT_UPWARD
		  || c == UNDERFLOW_EXCEPT_DOWNWARD);
	}
    }
  else
    return true;
}

static bool support_underflow_exception = false;
volatile double d = DBL_MIN;
volatile double dd;

static int
test_in_one_mode (const char *s, enum underflow_case c, int rm,
		  const char *mode_name)
{
  int result = 0;
  feclearexcept (FE_ALL_EXCEPT);
  errno = 0;
  double d = strtod (s, NULL);
  int got_errno = errno;
#ifdef FE_UNDERFLOW
  bool got_fe_underflow = fetestexcept (FE_UNDERFLOW) != 0;
#else
  bool got_fe_underflow = false;
#endif
  printf ("strtod (%s) (%s) returned %a, errno = %d, %sunderflow exception\n",
	  s, mode_name, d, got_errno, got_fe_underflow ? "" : "no ");
  bool this_expect_underflow = expect_underflow (c, rm);
  if (got_errno != 0 && got_errno != ERANGE)
    {
      puts ("FAIL: errno neither 0 nor ERANGE");
      result = 1;
    }
  else if (this_expect_underflow != (errno == ERANGE))
    {
      puts ("FAIL: underflow from errno differs from expectations");
      result = 1;
    }
  if (support_underflow_exception && got_fe_underflow != this_expect_underflow)
    {
      puts ("FAIL: underflow from exceptions differs from expectations");
      result = 1;
    }
  return result;
}

static int
do_test (void)
{
  int save_round_mode __attribute__ ((unused)) = fegetround ();
  int result = 0;
#ifdef FE_TONEAREST
  const int fe_tonearest = FE_TONEAREST;
#else
  const int fe_tonearest = 0;
# if defined FE_DOWNWARD || defined FE_TOWARDZERO || defined FE_UPWARD
#  error "FE_TONEAREST not defined, but another rounding mode is"
# endif
#endif
#ifdef FE_UNDERFLOW
  feclearexcept (FE_ALL_EXCEPT);
  dd = d * d;
  if (fetestexcept (FE_UNDERFLOW))
    support_underflow_exception = true;
  else
    puts ("underflow exception not supported at runtime, only testing errno");
#endif
  for (size_t i = 0; i < sizeof (tests) / sizeof (tests[0]); i++)
    {
      result |= test_in_one_mode (tests[i].s, tests[i].c, fe_tonearest,
				  "default rounding mode");
#ifdef FE_DOWNWARD
      if (!fesetround (FE_DOWNWARD))
	{
	  result |= test_in_one_mode (tests[i].s, tests[i].c, FE_DOWNWARD,
				      "FE_DOWNWARD");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_TOWARDZERO
      if (!fesetround (FE_TOWARDZERO))
	{
	  result |= test_in_one_mode (tests[i].s, tests[i].c, FE_TOWARDZERO,
				      "FE_TOWARDZERO");
	  fesetround (save_round_mode);
	}
#endif
#ifdef FE_UPWARD
      if (!fesetround (FE_UPWARD))
	{
	  result |= test_in_one_mode (tests[i].s, tests[i].c, FE_UPWARD,
				      "FE_UPWARD");
	  fesetround (save_round_mode);
	}
#endif
    }
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
