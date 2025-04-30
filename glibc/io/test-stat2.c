/* Test consistence of results of stat and stat64.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

int
main (int argc, char *argv[])
{
  int i;
  int result = 0;

  for (i = 1; i < argc; ++i)
    {
      struct stat st;
      struct stat64 st64;
      int same;

      if (stat (argv[i], &st) != 0)
	{
	  if (errno != EOVERFLOW)
	    {
	      /* Something is wrong.  */
	      printf ("stat(\"%s\",....) failed: %m", argv[i]);
	      result = 1;
	    }
	  continue;
	}

      if (stat64 (argv[i], &st64) != 0)
	{
	  if (errno != ENOSYS)
	    {
	      /* Something is wrong.  */
	      printf ("stat64(\"%s\",....) failed: %m", argv[i]);
	      result = 1;
	    }
	  continue;
	}

      printf ("\nName: %s\n", argv[i]);

#define TEST(name) \
      same = st.name == st64.name;					      \
      printf (#name ": %jd vs %jd  %s\n",				      \
	      (intmax_t) st.name, (intmax_t) st64.name,			      \
	      same ? "OK" : "FAIL");					      \
      result |= ! same

      TEST (st_dev);
      TEST (st_ino);
      TEST (st_mode);
      TEST (st_nlink);
      TEST (st_uid);
      TEST (st_gid);
#ifdef _STATBUF_ST_RDEV
      TEST (st_rdev);
#endif
#ifdef _STATBUF_ST_BLKSIZE
      TEST (st_blksize);
#endif
      TEST (st_blocks);
      TEST (st_atime);
      TEST (st_mtime);
      TEST (st_ctime);
    }

  return result;
}
