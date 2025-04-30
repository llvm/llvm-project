/* Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <libc-diag.h>

static int errors = 0;

static void
merror (const char *msg)
{
  ++errors;
  printf ("Error: %s\n", msg);
}

static int
do_test (void)
{
  void *volatile p;
  unsigned char *c;
  int save, i, ok;

  errno = 0;

  /* realloc (NULL, ...) behaves similarly to malloc (C89).  */
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about too-large allocations; here we want to test
     that they fail.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Walloc-size-larger-than=");
#endif
  p = realloc (NULL, -1);
  DIAG_POP_NEEDS_COMMENT;
  save = errno;

  if (p != NULL)
    merror ("realloc (NULL, -1) succeeded.");

  /* errno should be set to ENOMEM on failure (POSIX).  */
  if (p == NULL && save != ENOMEM)
    merror ("errno is not set correctly");

  errno = 0;

  /* realloc (NULL, ...) behaves similarly to malloc (C89).  */
  p = realloc (NULL, 10);
  save = errno;

  if (p == NULL)
    merror ("realloc (NULL, 10) failed.");

  free (p);

  p = calloc (20, 1);
  if (p == NULL)
    merror ("calloc (20, 1) failed.");

  /* Check increasing size preserves contents (C89).  */
  p = realloc (p, 200);
  if (p == NULL)
    merror ("realloc (p, 200) failed.");

  c = p;
  ok = 1;

  for (i = 0; i < 20; i++)
    {
      if (c[i] != 0)
        ok = 0;
    }

  if (ok == 0)
    merror ("first 20 bytes were not cleared");

  free (p);

  p = realloc (NULL, 100);
  if (p == NULL)
    merror ("realloc (NULL, 100) failed.");

  memset (p, 0xff, 100);

  /* Check decreasing size preserves contents (C89).  */
  p = realloc (p, 16);
  if (p == NULL)
    merror ("realloc (p, 16) failed.");

  c = p;
  ok = 1;

  for (i = 0; i < 16; i++)
    {
      if (c[i] != 0xff)
        ok = 0;
    }

  if (ok == 0)
    merror ("first 16 bytes were not correct");

  /* Check failed realloc leaves original untouched (C89).  */
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about too-large allocations; here we want to test
     that they fail.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Walloc-size-larger-than=");
#endif
  c = realloc (p, -1);
  DIAG_POP_NEEDS_COMMENT;
  if (c != NULL)
    merror ("realloc (p, -1) succeeded.");

  c = p;
  ok = 1;

  for (i = 0; i < 16; i++)
    {
      if (c[i] != 0xff)
        ok = 0;
    }

  if (ok == 0)
    merror ("first 16 bytes were not correct after failed realloc");

  /* realloc (p, 0) frees p (C89) and returns NULL (glibc).  */
  p = realloc (p, 0);
  if (p != NULL)
    merror ("realloc (p, 0) returned non-NULL.");

  /* realloc (NULL, 0) acts like malloc (0) (glibc).  */
  p = realloc (NULL, 0);
  if (p == NULL)
    merror ("realloc (NULL, 0) returned NULL.");

  free (p);

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
