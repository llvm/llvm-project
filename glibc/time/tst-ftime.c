/* Verify that ftime is sane.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <features.h>
#include <sys/timeb.h>
#include <libc-diag.h>

#include <support/check.h>

static int
do_test (void)
{
  struct timeb prev, curr = {.time = 0, .millitm = 0};
  int sec = 0;

  while (sec != 3)
    {
      prev = curr;

      /* ftime was deprecated on 2.31.  */
      DIAG_PUSH_NEEDS_COMMENT;
      DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

      TEST_COMPARE (ftime (&curr), 0);

      DIAG_POP_NEEDS_COMMENT;

      TEST_VERIFY (curr.time >= prev.time);

      if (curr.time == prev.time)
	TEST_VERIFY (curr.millitm >= prev.millitm);

      if (curr.time > prev.time)
        sec ++;
    }
  return 0;
}

#include <support/test-driver.c>
