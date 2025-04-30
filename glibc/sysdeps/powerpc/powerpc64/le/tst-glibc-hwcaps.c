/* glibc-hwcaps subdirectory test.  powerpc64le version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <support/check.h>
#include <sys/auxv.h>
#include <sys/param.h>

extern int marker2 (void);
extern int marker3 (void);

/* Return the POWER level, 8 for the baseline.  */
static int
compute_level (void)
{
  const char *platform = (const char *) getauxval (AT_PLATFORM);
  if (strcmp (platform, "power8") == 0)
    return 8;
  if (strcmp (platform, "power9") == 0)
    return 9;
  if (strcmp (platform, "power10") == 0)
    return 10;
  printf ("warning: unrecognized AT_PLATFORM value: %s\n", platform);
  /* Assume that the new platform supports POWER10.  */
  return 10;
}

static int
do_test (void)
{
  int level = compute_level ();
  printf ("info: detected POWER level: %d\n", level);
  TEST_COMPARE (marker2 (), MIN (level - 7, 2));
  TEST_COMPARE (marker3 (), MIN (level - 7, 3));
  return 0;
}

#include <support/test-driver.c>
