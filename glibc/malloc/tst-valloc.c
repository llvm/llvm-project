/* Test for valloc.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
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
  unsigned long pagesize = getpagesize ();
  unsigned long ptrval;
  int save;

  errno = 0;

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about too-large allocations; here we want to test
     that they fail.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Walloc-size-larger-than=");
#endif
  /* An attempt to allocate a huge value should return NULL and set
     errno to ENOMEM.  */
  p = valloc (-1);
#if __GNUC_PREREQ (7, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif

  save = errno;

  if (p != NULL)
    merror ("valloc (-1) succeeded.");

  if (p == NULL && save != ENOMEM)
    merror ("valloc (-1) errno is not set correctly");

  free (p);

  errno = 0;

  /* Test to expose integer overflow in malloc internals from BZ #15856.  */
  p = valloc (-pagesize);

  save = errno;

  if (p != NULL)
    merror ("valloc (-pagesize) succeeded.");

  if (p == NULL && save != ENOMEM)
    merror ("valloc (-pagesize) errno is not set correctly");

  free (p);

  /* A zero-sized allocation should succeed with glibc, returning a
     non-NULL value.  */
  p = valloc (0);

  if (p == NULL)
    merror ("valloc (0) failed.");

  free (p);

  /* Check the alignment of the returned pointer is correct.  */
  p = valloc (32);

  if (p == NULL)
    merror ("valloc (32) failed.");

  ptrval = (unsigned long) p;

  if ((ptrval & (pagesize - 1)) != 0)
    merror ("returned pointer is not page aligned.");

  free (p);

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
