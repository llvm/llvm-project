/* Test ldconfig does not segfault when aux-cache is corrupted (Bug 18093).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This test does the following:
   Run ldconfig to create the caches.
   Corrupt the caches.
   Run ldconfig again.
   At each step we verify that ldconfig does not crash.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <ftw.h>
#include <stdint.h>

#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>

#include <dirent.h>

static int
display_info (const char *fpath, const struct stat *sb,
              int tflag, struct FTW *ftwbuf)
{
  printf ("info: %-3s %2d %7jd   %-40s %d %s\n",
          (tflag == FTW_D) ? "d" : (tflag == FTW_DNR) ? "dnr" :
          (tflag == FTW_DP) ? "dp" : (tflag == FTW_F) ? "f" :
          (tflag == FTW_NS) ? "ns" : (tflag == FTW_SL) ? "sl" :
          (tflag == FTW_SLN) ? "sln" : "???",
          ftwbuf->level, (intmax_t) sb->st_size,
          fpath, ftwbuf->base, fpath + ftwbuf->base);
  /* To tell nftw to continue.  */
  return 0;
}

static void
execv_wrapper (void *args)
{
  char **argv = args;

  execv (argv[0], argv);
  FAIL_EXIT1 ("execv: %m");
}

/* Run ldconfig with a corrupt aux-cache, in particular we test for size
   truncation that might happen if a previous ldconfig run failed or if
   there were storage or power issues while we were writing the file.
   We want ldconfig not to crash, and it should be able to do so by
   computing the expected size of the file (bug 18093).  */
static int
do_test (void)
{
  char *prog = xasprintf ("%s/ldconfig", support_install_rootsbindir);
  char *args[] = { prog, NULL };
  const char *path = "/var/cache/ldconfig/aux-cache";
  struct stat64 fs;
  long int size, new_size, i;

  /* Create the needed directories. */
  xmkdirp ("/var/cache/ldconfig", 0777);

  /* Run ldconfig first to generate the aux-cache.  */
  struct support_capture_subprocess result;
  result = support_capture_subprocess (execv_wrapper, args);
  support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);
  support_capture_subprocess_free (&result);

  xstat (path, &fs);

  size = fs.st_size;
  /* Run 3 tests, each truncating aux-cache shorter and shorter.  */
  for (i = 3; i > 0; i--)
    {
      new_size = size * i / 4;
      if (truncate (path, new_size))
        FAIL_EXIT1 ("truncation failed: %m");
      if (nftw (path, display_info, 1000, 0) == -1)
        FAIL_EXIT1 ("nftw failed.");

      /* Verify that ldconfig can run with a truncated
         aux-cache and doesn't crash.  */
      struct support_capture_subprocess result;
      result = support_capture_subprocess (execv_wrapper, args);
      support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);
      support_capture_subprocess_free (&result);
    }

  free (prog);
  return 0;
}

#include <support/test-driver.c>
