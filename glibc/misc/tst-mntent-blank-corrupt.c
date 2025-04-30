/* Make sure blank lines does not cause memory corruption BZ #18887.

   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <mntent.h>
#include <stdio.h>
#include <string.h>

/* Make sure blank lines don't trigger memory corruption.  This doesn't happen
   for all targets though, so it's a best effort test BZ #18887.  */
static int
do_test (void)
{
  FILE *fp;

  fp = tmpfile ();
  fputs ("\n \n/foo\\040dir /bar\\040dir auto bind \t \n", fp);
  rewind (fp);

  /* The corruption happens here ...  */
  getmntent (fp);
  /* ... but trigers here.  */
  endmntent (fp);

  /* If the test failed, we would crash, and not hit this point.  */
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
