/* Test for ftw function related to symbolic links for BZ #23501
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

#include <ftw.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <support/support.h>
#include <support/check.h>

#define TSTDIR "tst-ftw-lnk.d"

static void
un (const char *file)
{
  struct stat st;
  /* Does the file exist?  */
  if (lstat (file, &st) < 0
      && errno == ENOENT)
    return;

  /* If so, try to remove it.  */
  if (unlink (file) < 0)
    FAIL_EXIT1 ("Unable to unlink %s", file);
}

static void
debug_cb (const char *which, const char *fpath,
	  const struct stat *sb, int typeflags)
{
  const char *sb_type = "???";
  const char *ftw_type = "???";

  /* Coding style here is intentionally "wrong" to increase readability.  */
  if (S_ISREG (sb->st_mode))  sb_type = "REG";
  if (S_ISDIR (sb->st_mode))  sb_type = "DIR";
  if (S_ISLNK (sb->st_mode))  sb_type = "LNK";

  if (typeflags == FTW_F)   ftw_type = "F";
  if (typeflags == FTW_D)   ftw_type = "D";
  if (typeflags == FTW_DNR) ftw_type = "DNR";
  if (typeflags == FTW_DP)  ftw_type = "DP";
  if (typeflags == FTW_NS)  ftw_type = "NS";
  if (typeflags == FTW_SL)  ftw_type = "SL";
  if (typeflags == FTW_SLN) ftw_type = "SLN";

  printf ("%s %5d %-3s %-3s %s\n", which, (int)(sb->st_ino % 100000), sb_type, ftw_type, fpath);
}

int good_cb = 0;
#define EXPECTED_GOOD 12

/* See if the stat buffer SB refers to the file AS_FNAME.  */
static void
check_same_stats (const struct stat *sb, const char *as_fname)
{
  struct stat as;
  if (lstat (as_fname, &as) < 0)
    FAIL_EXIT1 ("unable to stat %s for comparison", as_fname);

  if (as.st_mode == sb->st_mode
      && as.st_ino == sb->st_ino
      && as.st_size == sb->st_size)
    good_cb ++;
  else
    printf ("statbuf data doesn't match %s\n", as_fname);
}

static int
callback_phys (const char *fpath, const struct stat *sb, int typeflags, struct FTW *ftwbuf)
{
  debug_cb ("P", fpath, sb, typeflags);

  /* This callback is for when the FTW_PHYS flag is set.  The results
     should reflect the physical filesystem entry, not what it might
     point to.  */

  /* link1-bad is a dangling symlink, but we're reporting on the link
     anyway (ala lstat ()).  */
  if (strcmp (fpath, "./link1-bad") == 0)
    {
      if (S_ISLNK (sb->st_mode) && typeflags == FTW_SL)
	good_cb ++;
      else
	printf ("link1-bad had wrong phys stats\n");

      check_same_stats (sb, "link1-bad");
    }

  /* link2-ok is a regular non-dangling symlink.  */
  if (strcmp (fpath, "./link2-ok") == 0)
    {
      if (S_ISLNK (sb->st_mode) && typeflags == FTW_SL)
	good_cb ++;
      else
	printf ("link2-ok had wrong phys stats\n");

      check_same_stats (sb, "link2-ok");
    }

  /* This is the file link2-ok points to.  */
  if (strcmp (fpath, "./link2-tgt") == 0)
    {
      if (S_ISREG (sb->st_mode) && typeflags == FTW_F)
	good_cb ++;
      else
	printf ("link2-tgt had wrong phys stats\n");

      check_same_stats (sb, "link2-tgt");
    }

  return 0;
}

static int
callback_log (const char *fpath, const struct stat *sb, int typeflags, struct FTW *ftwbuf)
{
  debug_cb ("L", fpath, sb, typeflags);

  /* This callback is for when the FTW_PHYS flags is NOT set.  The
     results should reflect the logical file, i.e. symlinks should be
     followed.  */

  /* We would normally report what link1-bad links to, but link1-bad
     is a dangling symlink.  This is an exception to FTW_PHYS in that
     we report FTW_SLN (dangling symlink) but the stat data is
     correctly set to the link itself (ala lstat ()).  */
  if (strcmp (fpath, "./link1-bad") == 0)
    {
      if (S_ISLNK (sb->st_mode) && typeflags == FTW_SLN)
	good_cb ++;
      else
	printf ("link1-bad had wrong logical stats\n");

      check_same_stats (sb, "link1-bad");
    }

  /* link2-ok points to link2-tgt, so we expect data reflecting
     link2-tgt (ala stat ()).  */
  if (strcmp (fpath, "./link2-ok") == 0)
    {
      if (S_ISREG (sb->st_mode) && typeflags == FTW_F)
	good_cb ++;
      else
	printf ("link2-ok had wrong logical stats\n");

      check_same_stats (sb, "link2-tgt");
    }

  /* This is the file link2-ok points to.  */
  if (strcmp (fpath, "./link2-tgt") == 0)
    {
      if (S_ISREG (sb->st_mode) && typeflags == FTW_F)
	good_cb ++;
      else
	printf ("link2-tgt had wrong logical stats\n");

      check_same_stats (sb, "link2-tgt");
    }

  return 0;
}

static int
do_test (void)
{
  struct stat st;

  if (chdir (support_objdir_root) < 0)
    FAIL_EXIT1 ("cannot chdir to objdir root");

  if (chdir ("io") < 0)
    FAIL_EXIT1 ("cannot chdir to objdir/io subdir");

  if (stat (TSTDIR, &st) >= 0)
    {
      /* Directory does exist, delete any potential conflicts. */
      if (chdir (TSTDIR) < 0)
	FAIL_EXIT1 ("cannot chdir to %s\n", TSTDIR);
      un ("link1-bad");
      un ("link1-tgt");
      un ("link2-ok");
      un ("link2-tgt");
    }
  else
    {
      /* Directory does not exist, create it.  */
      mkdir (TSTDIR, 0777);
      if (chdir (TSTDIR) < 0)
	FAIL_EXIT1 ("cannot chdir to %s\n", TSTDIR);
    }

  /* At this point, we're inside our test directory, and need to
     prepare it.  */

  if (symlink ("link1-tgt", "link1-bad") < 0)
    FAIL_EXIT1 ("symlink link1-bad failed");
  if (symlink ("link2-tgt", "link2-ok") < 0)
    FAIL_EXIT1 ("symlink link2-ok failed");
  if (open ("link2-tgt", O_RDWR|O_CREAT, 0777) < 0)
    FAIL_EXIT1 ("create of link2-tgt failed");

  /* Now we run the tests.  */

  nftw (".", callback_phys, 10, FTW_PHYS);
  nftw (".", callback_log, 10, 0);

  /* Did we see the expected number of correct callbacks? */

  if (good_cb != EXPECTED_GOOD)
    {
      FAIL_EXIT1 ("Saw %d good callbacks, expected %d\n",
		  good_cb, EXPECTED_GOOD);
    }

  return 0;
}

#include <support/test-driver.c>
