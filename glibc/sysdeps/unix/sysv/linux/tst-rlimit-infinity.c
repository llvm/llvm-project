/* Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <sys/resource.h>
#include <support/check.h>

static int resources[] = {
  /* The following 7 limits are part of POSIX and must exist.  */
  RLIMIT_CORE,
  RLIMIT_CPU,
  RLIMIT_DATA,
  RLIMIT_FSIZE,
  RLIMIT_NOFILE,
  RLIMIT_STACK,
  RLIMIT_AS
};

#define nresources (sizeof (resources) / sizeof (resources[0]))

/* Assume that the prlimit64 function calls the prlimit64 syscall without
   mangling the arguments.  */
#define PRLIMIT64_INFINITY	((rlim64_t) -1)

/* As we don't know which limit will be modified, use a sufficiently high
   value to not shoot ourself in the foot.  Use a 32-bit value to test
   both the 32- and 64-bit versions, and keep the highest bit clear to
   avoid sign extension.  */
#define PRLIMIT64_TESTVAL	((rlim64_t) 0x42420000)

static void
test_getrlimit (int resource, rlim_t exp_cur, rlim_t exp_max)
{
  struct rlimit r;
  TEST_VERIFY_EXIT (getrlimit (resource, &r) == 0);
  TEST_COMPARE (r.rlim_cur, exp_cur);
  TEST_COMPARE (r.rlim_max, exp_max);
}

static void
test_getrlimit64 (int resource, rlim64_t exp_cur, rlim64_t exp_max)
{
  struct rlimit64 r;
  TEST_VERIFY_EXIT (getrlimit64 (resource, &r) == 0);
  TEST_COMPARE (r.rlim_cur, exp_cur);
  TEST_COMPARE (r.rlim_max, exp_max);
}

static void
test_prlimit_get (int resource, rlim_t exp_cur, rlim_t exp_max)
{
  struct rlimit r;
  TEST_VERIFY_EXIT (prlimit (0, resource, NULL, &r) == 0);
  TEST_COMPARE (r.rlim_cur, exp_cur);
  TEST_COMPARE (r.rlim_max, exp_max);
}

static void
test_prlimit64_get (int resource, rlim64_t exp_cur, rlim64_t exp_max)
{
  struct rlimit64 r;
  TEST_COMPARE (prlimit64 (0, resource, NULL, &r), 0);
  TEST_COMPARE (r.rlim_cur, exp_cur);
  TEST_COMPARE (r.rlim_max, exp_max);
}

static void
test_setrlimit (int resource, rlim_t new_cur, rlim_t new_max)
{
  struct rlimit r = { new_cur, new_max };
  TEST_COMPARE (setrlimit (resource, &r), 0);
}

static void
test_setrlimit64 (int resource, rlim64_t new_cur, rlim64_t new_max)
{
  struct rlimit64 r = { new_cur, new_max };
  TEST_COMPARE (setrlimit64 (resource, &r), 0);
}

static void
test_prlimit_set (int resource, rlim_t new_cur, rlim_t new_max)
{
  struct rlimit r = { new_cur, new_max };
  TEST_COMPARE (prlimit (0, resource, &r, NULL), 0);
}

static void
test_prlimit64_set (int resource, rlim64_t new_cur, rlim64_t new_max)
{
  struct rlimit64 r = { new_cur, new_max };
  TEST_COMPARE (prlimit64 (0, resource, &r, NULL), 0);
}

static int
do_test (void)
{
  int resource = -1;

  /* Find a resource with hard limit set to infinity, so that the soft limit
     can be manipulated to any value.  */
  for (int i = 0; i < nresources; ++i)
    {
      struct rlimit64 r64;
      int res = prlimit64 (0, resources[i], NULL, &r64);
      if ((res == 0) && (r64.rlim_max == PRLIMIT64_INFINITY))
	{
	  resource = resources[i];
	  break;
	}
    }

  if (resource == -1)
    FAIL_UNSUPPORTED
      ("Could not find and limit with hard limit set to infinity.");

  /* First check that the get functions work correctly with the test value.  */
  test_prlimit64_set (resource, PRLIMIT64_TESTVAL, PRLIMIT64_INFINITY);
  test_getrlimit (resource, PRLIMIT64_TESTVAL, RLIM_INFINITY);
  test_getrlimit64 (resource, PRLIMIT64_TESTVAL, RLIM64_INFINITY);
  test_prlimit_get (resource, PRLIMIT64_TESTVAL, RLIM_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_TESTVAL, RLIM64_INFINITY);

  /* Then check that the get functions work correctly with infinity.  */
  test_prlimit64_set (resource, PRLIMIT64_INFINITY, PRLIMIT64_INFINITY);
  test_getrlimit (resource, RLIM_INFINITY, RLIM_INFINITY);
  test_getrlimit64 (resource, RLIM64_INFINITY, RLIM64_INFINITY);
  test_prlimit_get (resource, RLIM_INFINITY, RLIM_INFINITY);
  test_prlimit64_get (resource, RLIM64_INFINITY, RLIM64_INFINITY);

  /* Then check that setrlimit works correctly with the test value.  */
  test_setrlimit (resource, PRLIMIT64_TESTVAL, RLIM_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_TESTVAL, PRLIMIT64_INFINITY);

  /* Then check that setrlimit works correctly with infinity.  */
  test_setrlimit (resource, RLIM_INFINITY, RLIM_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_INFINITY, PRLIMIT64_INFINITY);

  /* Then check that setrlimit64 works correctly with the test value.  */
  test_setrlimit64 (resource, PRLIMIT64_TESTVAL, RLIM64_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_TESTVAL, PRLIMIT64_INFINITY);

  /* Then check that setrlimit64 works correctly with infinity.  */
  test_setrlimit64 (resource, RLIM64_INFINITY, RLIM64_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_INFINITY, PRLIMIT64_INFINITY);

  /* Then check that prlimit works correctly with the test value.  */
  test_prlimit_set (resource, RLIM_INFINITY, RLIM_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_INFINITY, PRLIMIT64_INFINITY);

  /* Finally check that prlimit works correctly with infinity.  */
  test_prlimit_set (resource, PRLIMIT64_TESTVAL, RLIM_INFINITY);
  test_prlimit64_get (resource, PRLIMIT64_TESTVAL, PRLIMIT64_INFINITY);

  return 0;
}

#include <support/test-driver.c>
