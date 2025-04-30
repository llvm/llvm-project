/* Test support_record_failure state sharing.
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

#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int exit_status_with_failure = -1;
static bool test_verify;
static bool test_verify_exit;
enum
  {
    OPT_STATUS = 10001,
    OPT_TEST_VERIFY,
    OPT_TEST_VERIFY_EXIT,
  };
#define CMDLINE_OPTIONS                                                 \
  { "status", required_argument, NULL, OPT_STATUS },                    \
  { "test-verify", no_argument, NULL, OPT_TEST_VERIFY },                \
  { "test-verify-exit", no_argument, NULL, OPT_TEST_VERIFY_EXIT },
static void
cmdline_process (int c)
{
  switch (c)
    {
    case OPT_STATUS:
      exit_status_with_failure = atoi (optarg);
      break;
    case OPT_TEST_VERIFY:
      test_verify = true;
      break;
    case OPT_TEST_VERIFY_EXIT:
      test_verify_exit = true;
      break;
    }
}
#define CMDLINE_PROCESS cmdline_process

static void
check_failure_reporting (int phase, int zero, int unsupported)
{
  int status = support_report_failure (0);
  if (status != zero)
    {
      printf ("real-error (phase %d): support_report_failure (0) == %d\n",
              phase, status);
      exit (1);
    }
  status = support_report_failure (1);
  if (status != 1)
    {
      printf ("real-error (phase %d): support_report_failure (1) == %d\n",
              phase, status);
      exit (1);
    }
  status = support_report_failure (2);
  if (status != 2)
    {
      printf ("real-error (phase %d): support_report_failure (2) == %d\n",
              phase, status);
      exit (1);
    }
  status = support_report_failure (EXIT_UNSUPPORTED);
  if (status != unsupported)
    {
      printf ("real-error (phase %d): "
              "support_report_failure (EXIT_UNSUPPORTED) == %d\n",
              phase, status);
      exit (1);
    }
}

static int
do_test (void)
{
  if (exit_status_with_failure >= 0)
    {
      /* External invocation with requested error status.  Used by
         tst-support_report_failure-2.sh.  */
      support_record_failure ();
      return exit_status_with_failure;
    }
  TEST_VERIFY (true);
  TEST_VERIFY_EXIT (true);
  if (test_verify)
    {
      TEST_VERIFY (false);
      if (test_verbose)
        printf ("info: execution passed failed TEST_VERIFY\n");
      return 2; /* Expected exit status.  */
    }
  if (test_verify_exit)
    {
      TEST_VERIFY_EXIT (false);
      return 3; /* Not reached.  Expected exit status is 1.  */
    }

  printf ("info: This test tests the test framework.\n"
          "info: It reports some expected errors on stdout.\n");

  /* Check that the status is passed through unchanged.  */
  check_failure_reporting (1, 0, EXIT_UNSUPPORTED);

  /* Check state propagation from a subprocess.  */
  pid_t pid = xfork ();
  if (pid == 0)
    {
      support_record_failure ();
      _exit (0);
    }
  int status;
  xwaitpid (pid, &status, 0);
  if (status != 0)
    {
      printf ("real-error: incorrect status from subprocess: %d\n", status);
      return 1;
    }
  check_failure_reporting (2, 1, 1);

  /* Also test directly in the parent process.  */
  support_record_failure_reset ();
  check_failure_reporting (3, 0, EXIT_UNSUPPORTED);
  support_record_failure ();
  check_failure_reporting (4, 1, 1);

  /* We need to mask the failure above.  */
  support_record_failure_reset ();
  return 0;
}

#include <support/test-driver.c>
