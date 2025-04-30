/* Test for (lack of) command execution in wordexp.
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

/* This test optionally counts PIDs in a PID namespace to detect
   forks.  Without kernel support for that, it will merely look at the
   error codes from wordexp to check that no command execution
   happens.  */

#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xunistd.h>
#include <wordexp.h>

/* Set to true if the test runs in a PID namespace and can therefore
   use next_pid below.  */
static bool pid_tests_supported;

/* The next PID, as returned from next_pid below.  Only meaningful if
   pid_tests_supported.  */
static pid_t expected_pid;

/* Allocate the next PID and return it.  The process is terminated.
   Note that the test itself advances the next PID.  */
static pid_t
next_pid (void)
{
  pid_t pid = xfork ();
  if (pid == 0)
    _exit (0);
  xwaitpid (pid, NULL, 0);
  return pid;
}

/* Check that evaluating PATTERN with WRDE_NOCMD results in
   EXPECTED_ERROR.  */
static void
expect_failure (const char *pattern, int expected_error)
{
  printf ("info: testing pattern: %s\n", pattern);
  wordexp_t w;
  TEST_COMPARE (wordexp (pattern, &w, WRDE_NOCMD), expected_error);
  if (pid_tests_supported)
    TEST_COMPARE (expected_pid++, next_pid ());
}

/* Run all the tests.  Invoked with different IFS values.  */
static void
run_tests (void)
{
  /* Integer overflow in division.  */
  {
    static const char *const numbers[] = {
      "0",
      "1",
      "65536",
      "2147483648",
      "4294967296"
      "9223372036854775808",
      "18446744073709551616",
      "170141183460469231731687303715884105728",
      "340282366920938463463374607431768211456",
      NULL
    };

    for (const char *const *num = numbers; *num != NULL; ++num)
      {
        wordexp_t w;
        char pattern[256];
        snprintf (pattern, sizeof (pattern), "$[(-%s)/(-1)]", *num);
        int ret = wordexp (pattern, &w, WRDE_NOCMD);
        if (ret == 0)
          {
            /* If the call is successful, the result must match the
               original number.  */
            TEST_COMPARE (w.we_wordc, 1);
            TEST_COMPARE_STRING (w.we_wordv[0], *num);
            TEST_COMPARE_STRING (w.we_wordv[1], NULL);
            wordfree (&w);
          }
        else
          /* Otherwise, the test must fail with a syntax error.  */
          TEST_COMPARE (ret, WRDE_SYNTAX);

        /* In both cases, command execution is not permitted.  */
        if (pid_tests_supported)
          TEST_COMPARE (expected_pid++, next_pid ());
      }
  }

  /* (Lack of) command execution tests.  */

  expect_failure ("$(ls)", WRDE_CMDSUB);

  /* Test for CVE-2014-7817. We test 3 combinations of command
     substitution inside an arithmetic expression to make sure that
     no commands are executed and error is returned.  */
  expect_failure ("$((`echo 1`))", WRDE_CMDSUB);
  expect_failure ("$((1+`echo 1`))", WRDE_CMDSUB);
  expect_failure ("$((1+$((`echo 1`))))", WRDE_CMDSUB);

  expect_failure ("$[1/0]", WRDE_SYNTAX); /* BZ 18100.  */
}

static void
subprocess (void *closure)
{
  expected_pid = 2;
  if (pid_tests_supported)
    TEST_COMPARE (expected_pid++, next_pid ());

  /* Check that triggering command execution via wordexp results in a
     PID increase.  */
  if (pid_tests_supported)
    {
      wordexp_t w;
      TEST_COMPARE (wordexp ("$(echo Test)", &w, 0), 0);
      TEST_COMPARE (w.we_wordc, 1);
      TEST_COMPARE_STRING (w.we_wordv[0], "Test");
      TEST_COMPARE_STRING (w.we_wordv[1], NULL);
      wordfree (&w);

      pid_t n = next_pid ();
      printf ("info: self-test resulted in PID %d (processes created: %d)\n",
              (int) n, (int) (n - expected_pid));
      TEST_VERIFY (n > expected_pid);
      expected_pid = n + 1;
  }

  puts ("info: testing without IFS");
  unsetenv ("IFS");
  run_tests ();

  puts ("info: testing with IFS");
  TEST_COMPARE (setenv ("IFS", " \t\n", 1), 0);
  run_tests ();
}

static int
do_test (void)
{
  support_become_root ();

#ifdef CLONE_NEWPID
  if (unshare (CLONE_NEWPID) != 0)
    printf ("warning: unshare (CLONE_NEWPID) failed: %m\n"
            "warning: This leads to reduced test coverage.\n");
  else
    pid_tests_supported = true;
#else
  printf ("warning: CLONE_NEWPID not available.\n"
          "warning: This leads to reduced test coverage.\n");
#endif

  /* CLONE_NEWPID only has an effect after fork.  */
  support_isolate_in_subprocess (subprocess, NULL);

  return 0;
}

#include <support/test-driver.c>
