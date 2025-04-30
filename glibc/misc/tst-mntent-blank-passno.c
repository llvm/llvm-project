/* Make sure trailing whitespace is handled properly BZ #17273.

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

/* Check entries to make sure trailing whitespace is ignored and we return the
   correct passno value BZ #17273.  */
static int
do_test (void)
{
  int result = 0;
  FILE *fp;
  struct mntent *mnt;

  fp = tmpfile ();
  fputs ("/foo\\040dir /bar\\040dir auto bind \t \n", fp);
  rewind (fp);

  mnt = getmntent (fp);
  if (strcmp (mnt->mnt_fsname, "/foo dir") != 0
      || strcmp (mnt->mnt_dir, "/bar dir") != 0
      || strcmp (mnt->mnt_type, "auto") != 0
      || strcmp (mnt->mnt_opts, "bind") != 0
      || mnt->mnt_freq != 0
      || mnt->mnt_passno != 0)
    {
      puts ("Error while reading entry with trailing whitespaces");
      result = 1;
    }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
