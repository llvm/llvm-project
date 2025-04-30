/* Test that glibc.malloc.mxfast tunable works.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* This test verifies that setting the glibc.malloc.mxfast tunable to
   zero results in free'd blocks being returned to the small bins, not
   the fast bins.  */

#include <malloc.h>
#include <libc-diag.h>
#include <support/check.h>

int
do_test (void)
{
  struct mallinfo m;
  char *volatile p1;
  char *volatile p2;

  /* Arbitrary value; must be in default fastbin range.  */
  p1 = malloc (3);
  /* Something large so that p1 isn't a "top block" */
  p2 = malloc (512);
  free (p1);

  /* The test below covers the deprecated mallinfo function.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

  m = mallinfo ();

  DIAG_POP_NEEDS_COMMENT;

  /* This will fail if there are any blocks in the fastbins.  */
  TEST_COMPARE (m.smblks, 0);

  /* To keep gcc happy.  */
  free (p2);

  return 0;
}

#include <support/test-driver.c>
