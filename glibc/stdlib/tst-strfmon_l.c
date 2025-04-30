/* Test locale dependence of strfmon_l.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <stdio.h>
#include <monetary.h>
#include <string.h>
#include <stdlib.h>
#include <locale.h>

static const char *const en_us_name = "en_US.ISO-8859-1";

/* Locale value to be used by tests.  */
static locale_t loc;
static const char *loc_name;

/* Set the global locale to GLOBAL_NAME, and the locale referenced by
   the loc variable above to LOCAL_NAME.  */
static void
init_loc (const char *global_name, const char *local_name)
{
  loc = newlocale (LC_ALL_MASK, local_name, 0);
  if (loc == 0)
    {
      printf ("error: newlocale (%s): %m\n", local_name);
      abort ();
    }
  loc_name = local_name;

  if (setlocale (LC_ALL, global_name) == NULL)
    {
      printf ("error: setlocale (%s): %m\n", global_name);
      abort ();
    }
}

/* Expected strings for a positive or negative value.  */
struct testcase
{
  const char *i;                /* %i */
  const char *n;                /* %n */
  const char *i_ungrouped;      /* %^i */
  const char *n_ungrouped;      /* %^n */
};

/* Collected expected strings for both positive and negative
   values.  */
struct testcase_pair
{
  struct testcase positive;     /* 1234567.89 */
  struct testcase negative;     /* -1234567.89 */
};

static bool errors;

/* Test one value using the locale loc.  */
static void
test_one (const char *format, double value, const char *ldformat,
	  long double ldvalue, const char *expected)
{
  static char actual[64], actualld[64];
  int result = strfmon_l (actual, sizeof (actual), loc, format, value);
  int res_ld = strfmon_l (actualld, sizeof (actualld), loc, ldformat, ldvalue);
  if (result < 0)
    {
      printf ("error: locale %s, format \"%s\", value %g: strfmon_l: %m\n",
              loc_name, format, value);
      errors = true;
    }
  else if (res_ld < 0)
    {
      printf ("error: locale %s, format \"%s\", value %Lg: strfmon_l: %m\n",
	      loc_name, ldformat, ldvalue);
      errors = true;
    }
  else if (strcmp (actual, expected) != 0)
    {
      printf ("error: locale %s, format \"%s\", value %g: mismatch\n",
              loc_name, format, value);
      printf ("error:   expected: \"%s\"\n", expected);
      printf ("error:   actual:   \"%s\"\n", actual);
      errors = true;
    }
  else if (strcmp (actualld, expected) != 0)
    {
      printf ("error: locale %s, format \"%s\", value %Lg: mismatch\n",
	      loc_name, ldformat, ldvalue);
      printf ("error:   expected: \"%s\"\n", expected);
      printf ("error:   actual:   \"%s\"\n", actualld);
      errors = true;
    }
}

static void
test_pair (const struct testcase_pair *pair)
{
  double positive = 1234567.89;
  long double pos = 1234567.89L;
  test_one ("%i", positive, "%Li", pos, pair->positive.i);
  test_one ("%n", positive, "%Ln", pos, pair->positive.n);
  test_one ("%^i", positive, "%^Li", pos, pair->positive.i_ungrouped);
  test_one ("%^n", positive, "%^Ln", pos, pair->positive.n_ungrouped);
  double negative = -1234567.89;
  long double neg = -1234567.89L;
  test_one ("%i", negative, "%Li", neg, pair->negative.i);
  test_one ("%n", negative, "%Ln", neg, pair->negative.n);
  test_one ("%^i", negative, "%^Li", neg, pair->negative.i_ungrouped);
  test_one ("%^n", negative, "%^Ln", neg, pair->negative.n_ungrouped);
}

static const struct testcase_pair en_us =
  {
    {
      "USD 1,234,567.89", "$1,234,567.89",
      "USD 1234567.89", "$1234567.89"
    },
    {
      "-USD 1,234,567.89", "-$1,234,567.89",
      "-USD 1234567.89", "-$1234567.89"
    }
  };

static void
test_en_us (const char *other_name)
{
  init_loc (other_name, en_us_name);
  test_pair (&en_us);
  freelocale (loc);
}

struct locale_pair
{
  const char *locale_name;
  struct testcase_pair pair;
};

static const struct locale_pair tests[] =
  {
    {
      "de_DE.UTF-8",
      {
        {
         "1.234.567,89 EUR", "1.234.567,89 \u20ac",
         "1234567,89 EUR", "1234567,89 \u20ac"
        },
        {
         "-1.234.567,89 EUR", "-1.234.567,89 \u20ac",
         "-1234567,89 EUR", "-1234567,89 \u20ac"
        }
      },
    },
    {
      "tg_TJ.UTF-8",
      {
        {
          "1\u202f234\u202f567.89 TJS", "1\u202f234\u202f567.89 \u0440\u0443\u0431",
          "1234567.89 TJS", "1234567.89 \u0440\u0443\u0431"
        },
        {
          "-1\u202f234\u202f567.89 TJS", "-1\u202f234\u202f567.89 \u0440\u0443\u0431",
          "-1234567.89 TJS", "-1234567.89 \u0440\u0443\u0431"
        }
      }
    },
    {
      "hr_HR.UTF-8",
      {
        {
          "HRK 1.234.567,89", "1.234.567,89 kn",
          "HRK 1234567,89", "1234567,89 kn"
        },
        {
          "-HRK 1.234.567,89", "-1.234.567,89 kn",
          "-HRK 1234567,89", "-1234567,89 kn"
        }
      }
    },
    {
      "hi_IN.UTF-8",
      {
        {
          "INR12,34,567.89", "\u20b912,34,567.89",
          "INR1234567.89", "\u20b91234567.89"
        },
        {
          "-INR12,34,567.89", "-\u20b912,34,567.89",
          "-INR1234567.89", "-\u20b91234567.89"
        }
      }
    },
    {
      "el_GR.UTF-8",
      {
        {
          "1.234.567,89EUR", "1.234.567,89\u20ac",
          "1234567,89EUR", "1234567,89\u20ac"
        },
        {
          "-1.234.567,89EUR", "-1.234.567,89\u20ac",
          "-1234567,89EUR", "-1234567,89\u20ac",
        }
      }
    },
    {}
  };

static int
do_test (void)
{
  for (const struct locale_pair *test = tests;
       test->locale_name != NULL; ++test)
    {
      init_loc (en_us_name, test->locale_name);
      test_pair (&test->pair);
      freelocale (loc);
      test_en_us (test->locale_name);
    }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
