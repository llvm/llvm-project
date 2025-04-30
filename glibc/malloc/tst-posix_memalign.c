/* Test for posix_memalign.
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
  void *p;
  int ret;
  unsigned long pagesize = getpagesize ();
  unsigned long ptrval;

  p = NULL;

  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  /* GCC 7 warns about too-large allocations; here we want to test
     that they fail.  */
  DIAG_IGNORE_NEEDS_COMMENT (7, "-Walloc-size-larger-than=");
#endif
  /* An attempt to allocate a huge value should return ENOMEM and
     p should remain NULL.  */
  ret = posix_memalign (&p, sizeof (void *), -1);
#if __GNUC_PREREQ (7, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif

  if (ret != ENOMEM)
    merror ("posix_memalign (&p, sizeof (void *), -1) succeeded.");

  if (ret == ENOMEM && p != NULL)
    merror ("returned an error but pointer was modified");

  free (p);

  p = NULL;

  /* Test to expose integer overflow in malloc internals from BZ #15857.  */
  ret = posix_memalign (&p, pagesize, -pagesize);

  if (ret != ENOMEM)
    merror ("posix_memalign (&p, pagesize, -pagesize) succeeded.");

  free (p);

  p = NULL;

  /* Test to expose integer overflow in malloc internals from BZ #16038.  */
  ret = posix_memalign (&p, -1, pagesize);

  if (ret != EINVAL)
    merror ("posix_memalign (&p, -1, pagesize) succeeded.");

  free (p);

  p = NULL;

  /* A zero-sized allocation should succeed with glibc, returning zero
     and setting p to a non-NULL value.  */
  ret = posix_memalign (&p, sizeof (void *), 0);

  if (ret != 0 || p == NULL)
    merror ("posix_memalign (&p, sizeof (void *), 0) failed.");

  free (p);

  ret = posix_memalign (&p, 0x300, 10);

  if (ret != EINVAL)
    merror ("posix_memalign (&p, 0x300, 10) succeeded.");

  ret = posix_memalign (&p, 0, 10);

  if (ret != EINVAL)
    merror ("posix_memalign (&p, 0, 10) succeeded.");

  p = NULL;

  ret = posix_memalign (&p, 0x100, 10);

  if (ret != 0)
    merror ("posix_memalign (&p, 0x100, 10) failed.");

  if (ret == 0 && p == NULL)
    merror ("returned success but pointer is NULL");

  ptrval = (unsigned long) p;

  if (ret == 0 && (ptrval & 0xff) != 0)
    merror ("pointer is not aligned to 0x100");

  free (p);

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
