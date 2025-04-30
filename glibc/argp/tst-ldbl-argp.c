/* Testing of long double conversions in argp.h functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <argp.h>
#include <string.h>

#include <support/capture_subprocess.h>
#include <support/check.h>

static const struct argp_option
options[] =
{
  { "error", 'e', "format", OPTION_ARG_OPTIONAL,
    "Run argp_error function with a format string", 0 },
  { "failure", 'f', "format", OPTION_ARG_OPTIONAL,
    "Run argp_failure function with a format string", 0 },
  { NULL, 0, NULL, 0, NULL }
};

static error_t
parser (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
      case 'e':
	argp_error (state, "%Lf%f%Lf%f", (long double) -1, (double) -2,
		    (long double) -3, (double) -4);
	break;
      case 'f':
	argp_failure (state, 0, 0, "%Lf%f%Lf%f", (long double) -1,
		      (double) -2, (long double) -3, (double) -4);
	break;
      default:
	return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

static struct argp
argp =
{
  options, parser
};

int argc = 2;
char *argv[3] = { (char *) "test-argp", NULL, NULL };

static void
do_test_call (void)
{
  int remaining;
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);
}

static int
do_one_test (const char *expected)
{
  struct support_capture_subprocess result;
  result = support_capture_subprocess ((void *) &do_test_call, NULL);

  TEST_COMPARE_STRING (result.err.buffer, expected);

  return 0;
}

static int
do_test (void)
{
  const char *param_error = "--error";
  const char *expected_error =
    "test-argp: -1.000000-2.000000-3.000000-4.000000\n"
    "Try `test-argp --help' or `test-argp --usage' for more information.\n";

  const char *param_failure = "--failure";
  const char *expected_failure =
    "test-argp: -1.000000-2.000000-3.000000-4.000000\n";

  argv[1] = (char *) param_error;
  do_one_test (expected_error);

  argv[1] = (char *) param_failure;
  do_one_test (expected_failure);

  return 0;
}

#include <support/test-driver.c>
