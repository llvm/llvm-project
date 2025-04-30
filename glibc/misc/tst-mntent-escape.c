/* Test mntent interface with escaped sequences.
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

#include <mntent.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>

struct const_mntent
{
  const char *mnt_fsname;
  const char *mnt_dir;
  const char *mnt_type;
  const char *mnt_opts;
  int mnt_freq;
  int mnt_passno;
  const char *expected;
};

struct const_mntent tests[] =
{
    {"/dev/hda1", "/some dir", "ext2", "defaults", 1, 2,
     "/dev/hda1 /some\\040dir ext2 defaults 1 2\n"},
    {"device name", "/some dir", "tmpfs", "defaults", 1, 2,
     "device\\040name /some\\040dir tmpfs defaults 1 2\n"},
    {" ", "/some dir", "tmpfs", "defaults", 1, 2,
     "\\040 /some\\040dir tmpfs defaults 1 2\n"},
    {"\t", "/some dir", "tmpfs", "defaults", 1, 2,
     "\\011 /some\\040dir tmpfs defaults 1 2\n"},
    {"\\", "/some dir", "tmpfs", "defaults", 1, 2,
     "\\134 /some\\040dir tmpfs defaults 1 2\n"},
};

static int
do_test (void)
{
  for (int i = 0; i < sizeof (tests) / sizeof (struct const_mntent); i++)
    {
      char buf[128];
      struct mntent *ret, curtest;
      FILE *fp = fmemopen (buf, sizeof (buf), "w+");

      if (fp == NULL)
	{
	  printf ("Failed to open file\n");
	  return 1;
	}

      curtest.mnt_fsname = strdupa (tests[i].mnt_fsname);
      curtest.mnt_dir = strdupa (tests[i].mnt_dir);
      curtest.mnt_type = strdupa (tests[i].mnt_type);
      curtest.mnt_opts = strdupa (tests[i].mnt_opts);
      curtest.mnt_freq = tests[i].mnt_freq;
      curtest.mnt_passno = tests[i].mnt_passno;

      if (addmntent (fp, &curtest) != 0)
	{
	  support_record_failure ();
	  continue;
	}

      TEST_COMPARE_STRING (buf, tests[i].expected);

      rewind (fp);
      ret = getmntent (fp);
      if (ret == NULL)
	{
	  support_record_failure ();
	  continue;
	}

      TEST_COMPARE_STRING(tests[i].mnt_fsname, ret->mnt_fsname);
      TEST_COMPARE_STRING(tests[i].mnt_dir, ret->mnt_dir);
      TEST_COMPARE_STRING(tests[i].mnt_type, ret->mnt_type);
      TEST_COMPARE_STRING(tests[i].mnt_opts, ret->mnt_opts);
      TEST_COMPARE(tests[i].mnt_freq, ret->mnt_freq);
      TEST_COMPARE(tests[i].mnt_passno, ret->mnt_passno);

      fclose (fp);
    }

  return 0;
}

#include <support/test-driver.c>
