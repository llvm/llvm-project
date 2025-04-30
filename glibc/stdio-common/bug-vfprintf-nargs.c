/* Test for vfprintf nargs allocation overflow (BZ #13656).
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kees Cook <keescook@chromium.org>, 2012.

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
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <signal.h>

static int
format_failed (const char *fmt, const char *expected)
{
  char output[80];

  printf ("%s : ", fmt);

  memset (output, 0, sizeof output);
  /* Having sprintf itself detect a failure is good.  */
  if (sprintf (output, fmt, 1, 2, 3, "test") > 0
      && strcmp (output, expected) != 0)
    {
      printf ("FAIL (output '%s' != expected '%s')\n", output, expected);
      return 1;
    }
  puts ("ok");
  return 0;
}

static int
do_test (void)
{
  int rc = 0;
  char buf[64];

  /* Regular positionals work.  */
  if (format_failed ("%1$d", "1") != 0)
    rc = 1;

  /* Regular width positionals work.  */
  if (format_failed ("%1$*2$d", " 1") != 0)
    rc = 1;

  /* Positional arguments are constructed via read_int, so nargs can only
     overflow on 32-bit systems.  On 64-bit systems, it will attempt to
     allocate a giant amount of memory and possibly crash, which is the
     expected situation.  Since the 64-bit behavior is arch-specific, only
     test this on 32-bit systems.  */
  if (sizeof (long int) == 4)
    {
      sprintf (buf, "%%1$d %%%" PRIdPTR "$d",
	       (intptr_t) (UINT32_MAX / sizeof (int)));
      if (format_failed (buf, "1 %$d") != 0)
        rc = 1;
    }

  return rc;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
