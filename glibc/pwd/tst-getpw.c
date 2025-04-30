/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <pwd.h>
#include <errno.h>
#include <stdbool.h>

/* We want to test getpw by calling it with a uid that does
   exist and one that doesn't exist. We track if we've met those
   conditions and exit. We also track if we've failed due to lack
   of memory. That constitutes all of the standard failure cases.  */
bool seen_hit;
bool seen_miss;
bool seen_oom;

/* How many errors we've had while running the test.  */
int errors;

static void
check (uid_t uid)
{
  int ret;
  char buf[1024];

  ret = getpw (uid, buf);

  /* Successfully read a password line.  */
  if (ret == 0 && !seen_hit)
    {
      printf ("PASS: Read a password line given a uid.\n");
      seen_hit = true;
    }

  /* Failed to read a password line. Why?  */
  if (ret == -1)
    {
      /* No entry?  Technically the errno could be any number
	 of values including ESRCH, EBADP or EPERM depending
	 on the quality of the nss module that implements the
	 underlying lookup. It should be 0 for getpw.*/
      if (errno == 0 && !seen_miss)
	{
	  printf ("PASS: Found an invalid uid.\n");
	  seen_miss = true;
	  return;
	}

      /* Out of memory?  */
      if (errno == ENOMEM && !seen_oom)
	{
	  printf ("FAIL: Failed with ENOMEM.\n");
	  seen_oom = true;
	  errors++;
	}

      /* We don't expect any other values for errno.  */
      if (errno != ENOMEM && errno != 0)
	errors++;
    }
}

static int
do_test (void)
{
  int ret;
  uid_t uid;

  /* Should return -1 and set errnot to EINVAL.  */
  ret = getpw (0, NULL);
  if (ret == -1 && errno == EINVAL)
    {
      printf ("PASS: NULL buffer returns -1 and sets errno to EINVAL.\n");
    }
  else
    {
      printf ("FAIL: NULL buffer did not return -1 or set errno to EINVAL.\n");
      errors++;
    }

  /* Look for one matching uid, one non-found uid and then stop.
     Set an upper limit at the 16-bit UID mark; no need to go farther.  */
  for (uid = 0; uid < ((uid_t) 65535); ++uid)
    {
      check (uid);
      if (seen_miss && seen_hit)
	break;
    }

  if (!seen_hit)
    printf ("FAIL: Did not read even one password line given a uid.\n");

  if (!seen_miss)
    printf ("FAIL: Did not find even one invalid uid.\n");

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
