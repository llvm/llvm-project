/* Test for fmemopen implementation.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Hanno Mueller, kontakt@hanno.de, 2000.

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

static char buffer[] = "foobar";

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <mcheck.h>

static int
do_bz18820 (void)
{
  char ch;
  FILE *stream;

  errno = 0;
  stream = fmemopen (&ch, 1, "?");
  if (stream)
    {
      printf ("fmemopen: expected NULL, got %p\n", stream);
      fclose (stream);
      return 1;
    }
  if (errno != EINVAL)
    {
      printf ("fmemopen: got %i, expected EINVAL (%i)\n", errno, EINVAL);
      return 10;
    }

  stream = fmemopen (NULL, 42, "?");
  if (stream)
    {
      printf ("fmemopen: expected NULL, got %p\n", stream);
      fclose (stream);
      return 2;
    }

  errno = 0;
  stream = fmemopen (NULL, ~0, "w");
  if (stream)
    {
      printf ("fmemopen: expected NULL, got %p\n", stream);
      fclose (stream);
      return 3;
    }
  if (errno != ENOMEM)
    {
      printf ("fmemopen: got %i, expected ENOMEM (%i)\n", errno, ENOMEM);
      return 20;
    }

  return 0;
}

static int
do_test (void)
{
  int ch;
  FILE *stream;
  int ret = 0;

  mtrace ();

  stream = fmemopen (buffer, strlen (buffer), "r+");

  while ((ch = fgetc (stream)) != EOF)
    printf ("Got %c\n", ch);

  fputc ('1', stream);
  if (fflush (stream) != EOF || errno != ENOSPC)
    {
      printf ("fflush didn't fail with ENOSPC\n");
      ret = 1;
    }

  fclose (stream);

  return ret + do_bz18820 ();
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
