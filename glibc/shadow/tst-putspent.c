/* Test for processing of invalid shadow entries.  [BZ #18724]
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

#include <errno.h>
#include <shadow.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool errors;

static void
check (struct spwd p, const char *expected)
{
  char *buf;
  size_t buf_size;
  FILE *f = open_memstream (&buf, &buf_size);

  if (f == NULL)
    {
      printf ("open_memstream: %m\n");
      errors = true;
      return;
    }

  int ret = putspent (&p, f);

  if (expected == NULL)
    {
      if (ret == -1)
	{
	  if (errno != EINVAL)
	    {
	      printf ("putspent: unexpected error code: %m\n");
	      errors = true;
	    }
	}
      else
	{
	  printf ("putspent: unexpected success (\"%s\")\n", p.sp_namp);
	  errors = true;
	}
    }
  else
    {
      /* Expect success.  */
      size_t expected_length = strlen (expected);
      if (ret == 0)
	{
	  long written = ftell (f);

	  if (written <= 0 || fflush (f) < 0)
	    {
	      printf ("stream error: %m\n");
	      errors = true;
	    }
	  else if (buf[written - 1] != '\n')
	    {
	      printf ("FAILED: \"%s\" without newline\n", expected);
	      errors = true;
	    }
	  else if (strncmp (buf, expected, written - 1) != 0
		   || written - 1 != expected_length)
	    {
	      printf ("FAILED: \"%s\" (%ld), expected \"%s\" (%zu)\n",
		      buf, written - 1, expected, expected_length);
	      errors = true;
	    }
	}
      else
	{
	  printf ("FAILED: putspent (expected \"%s\"): %m\n", expected);
	  errors = true;
	}
    }

  fclose (f);
  free (buf);
}

static int
do_test (void)
{
  check ((struct spwd) {
      .sp_namp = (char *) "root",
    },
    "root::0:0:0:0:0:0:0");
  check ((struct spwd) {
      .sp_namp = (char *) "root",
      .sp_pwdp = (char *) "password",
    },
    "root:password:0:0:0:0:0:0:0");
  check ((struct spwd) {
      .sp_namp = (char *) "root",
      .sp_pwdp = (char *) "password",
      .sp_lstchg = -1,
      .sp_min = -1,
      .sp_max = -1,
      .sp_warn = -1,
      .sp_inact = -1,
      .sp_expire = -1,
      .sp_flag = -1
    },
    "root:password:::::::");
  check ((struct spwd) {
      .sp_namp = (char *) "root",
      .sp_pwdp = (char *) "password",
      .sp_lstchg = 1,
      .sp_min = 2,
      .sp_max = 3,
      .sp_warn = 4,
      .sp_inact = 5,
      .sp_expire = 6,
      .sp_flag = 7
    },
    "root:password:1:2:3:4:5:6:7");

  /* Bad values.  */
  {
    static const char *const bad_strings[] = {
      ":",
      "\n",
      ":bad",
      "\nbad",
      "b:ad",
      "b\nad",
      "bad:",
      "bad\n",
      "b:a\nd",
      NULL
    };
    for (const char *const *bad = bad_strings; *bad != NULL; ++bad)
      {
	check ((struct spwd) {
	    .sp_namp = (char *) *bad,
	  }, NULL);
	check ((struct spwd) {
	    .sp_namp = (char *) "root",
	    .sp_pwdp = (char *) *bad,
	  }, NULL);
      }
  }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
