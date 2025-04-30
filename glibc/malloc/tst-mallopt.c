/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <malloc.h>
#include <stdio.h>
#include <string.h>

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
  int ret;

  ret = mallopt(M_CHECK_ACTION, 1);

  if (ret != 1)
    merror ("mallopt (M_CHECK_ACTION, 1) failed.");

  ret = mallopt(M_MMAP_MAX, 64*1024);

  if (ret != 1)
    merror ("mallopt (M_MMAP_MAX, 64*1024) failed.");

  ret = mallopt(M_MMAP_THRESHOLD, 64*1024);

  if (ret != 1)
    merror ("mallopt (M_MMAP_THRESHOLD, 64*1024) failed.");

  ret = mallopt(M_MXFAST, 0);

  if (ret != 1)
    merror ("mallopt (M_MXFAST, 0) failed.");

  ret = mallopt(M_PERTURB, 0xa5);

  if (ret != 1)
    merror ("mallopt (M_PERTURB, 0xa5) failed.");

  ret = mallopt(M_TOP_PAD, 64*1024);

  if (ret != 1)
    merror ("mallopt (M_TOP_PAD, 64*1024) failed.");

  ret = mallopt(M_TRIM_THRESHOLD, -1);

  if (ret != 1)
    merror ("mallopt (M_TRIM_THRESHOLD, -1) failed.");

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
