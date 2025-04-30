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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libc-diag.h>

#define VAR "FOOBAR"

char putenv_val[100] = VAR "=some longer value";

static int
do_test (void)
{
  int result = 0;
  const char *valp;

  /* First test: remove entry FOOBAR, whether it exists or not.  */
  unsetenv (VAR);

  /* Now getting the value should fail.  */
  if (getenv (VAR) != NULL)
    {
      printf ("There should be no `%s' value\n", VAR);
      result = 1;
    }

  /* Now add a value, with the replace flag cleared.  */
  if (setenv (VAR, "one", 0) != 0)
    {
      printf ("setenv #1 failed: %m\n");
      result = 1;
    }

  /* Getting this value should now be possible.  */
  valp = getenv (VAR);
  if (valp == NULL || strcmp (valp, "one") != 0)
    {
      puts ("getenv #2 failed");
      result = 1;
    }

  /* Try to replace without the replace flag set.  This should fail.  */
  if (setenv (VAR, "two", 0) != 0)
    {
      printf ("setenv #2 failed: %m\n");
      result = 1;
    }

  /* The value shouldn't have changed.  */
  valp = getenv (VAR);
  if (valp == NULL || strcmp (valp, "one") != 0)
    {
      puts ("getenv #3 failed");
      result = 1;
    }

  /* Now replace the value using putenv.  */
  if (putenv (putenv_val) != 0)
    {
      printf ("putenv #1 failed: %m\n");
      result = 1;
    }

  /* The value should have changed now.  */
  valp = getenv (VAR);
  if (valp == NULL || strcmp (valp, "some longer value") != 0)
    {
      printf ("getenv #4 failed (is \"%s\")\n", valp);
      result = 1;
    }

  /* Now one tricky check: changing the variable passed in putenv should
     change the environment.  */
  strcpy (&putenv_val[sizeof VAR], "a short one");

  /* The value should have changed again.  */
  valp = getenv (VAR);
  if (valp == NULL || strcmp (valp, "a short one") != 0)
    {
      puts ("getenv #5 failed");
      result = 1;
    }

  /* It should even be possible to rename the variable.  */
  strcpy (putenv_val, "XYZZY=some other value");

  /* Now a lookup using the old name should fail.  */
  if (getenv (VAR) != NULL)
    {
      puts ("getenv #6 failed");
      result = 1;
    }

  /* But using the new name it should work.  */
  valp = getenv ("XYZZY");
  if (valp == NULL || strcmp (valp, "some other value") != 0)
    {
      puts ("getenv #7 failed");
      result = 1;
    }

  /* Create a new variable with the old name.  */
  if (setenv (VAR, "a new value", 0) != 0)
    {
      printf ("setenv #3 failed: %m\n");
      result = 1;
    }

  /* At this point a getenv call must return the new value.  */
  valp = getenv (VAR);
  if (valp == NULL || strcmp (valp, "a new value") != 0)
    {
      puts ("getenv #8 failed");
      result = 1;
    }

  /* Black magic: rename the variable we added using putenv back.  */
  strcpy (putenv_val, VAR "=old name new value");

  /* This is interesting.  We have two variables with the same name.
     Getting a value should return one of them.  */
  valp = getenv (VAR);
  if (valp == NULL
      || (strcmp (valp, "a new value") != 0
	  && strcmp (valp, "old name new value") != 0))
    {
      puts ("getenv #9 failed");
      result = 1;
    }

  /* More fun ahead: we are now removing the variable.  This should remove
     both values.  The cast is ok: this call should never put the string
     in the environment and it should never modify it.  */
  putenv ((char *) VAR);

  /* Getting the value should now fail.  */
  if (getenv (VAR) != NULL)
    {
      printf ("getenv #10 failed (\"%s\" found)\n", getenv (VAR));
      result = 1;
    }

  /* Now a test with an environment variable that's one character long.
     This is to test a special case in the getenv implementation.  */
  strcpy (putenv_val, "X=one character test");
  if (putenv (putenv_val) != 0)
    {
      printf ("putenv #2 failed: %m\n");
      result = 1;
    }

  valp = getenv ("X");
  if (valp == NULL || strcmp (valp, "one character test") != 0)
    {
      puts ("getenv #11 failed");
      result = 1;
    }

  /* Both setenv and unsetenv should return -1/EINVAL for NULL or "" name
     or if name contains '=' character.  */
  errno = 0;
  if (setenv (NULL, "val", 1) >= 0 || errno != EINVAL)
    {
      puts ("setenv #4 failed");
      result = 1;
    }

  errno = 0;
  if (setenv ("", "val", 0) >= 0 || errno != EINVAL)
    {
      puts ("setenv #5 failed");
      result = 1;
    }

  errno = 0;
  if (setenv ("var=val", "val", 1) >= 0 || errno != EINVAL)
    {
      puts ("setenv #6 failed");
      result = 1;
    }

  /* This deliberately tests supplying a null pointer to a function whose
     argument is marked __attribute__ ((nonnull)). */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT(5, "-Wnonnull");
  errno = 0;
  if (unsetenv (NULL) >= 0 || errno != EINVAL)
    {
      puts ("unsetenv #1 failed");
      result = 1;
    }
  DIAG_POP_NEEDS_COMMENT;

  errno = 0;
  if (unsetenv ("") >= 0 || errno != EINVAL)
    {
      puts ("unsetenv #2 failed");
      result = 1;
    }

  errno = 0;
  if (unsetenv ("x=y") >= 0 || errno != EINVAL)
    {
      puts ("unsetenv #3 failed");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
