/* Test glob danglin symlink match (BZ #866).
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <stddef.h>
#include <glob.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <support/check.h>
#include <support/temp_file.h>

static void do_prepare (int argc, char *argv[]);
#define PREPARE do_prepare
static int do_test (void);
#include <support/test-driver.c>

/* Maximum number of symlink calls for create_link function.  */
#define MAX_CREATE_LINK_TRIES 10

static void
create_link (const char *base, const char *fname, char *linkname,
	     size_t linknamesize)
{
  int ntries = 0;
  while (1)
    {
      snprintf (linkname, linknamesize, "%s/%s%02d", test_dir, base,
		ntries);
      if (symlink (fname, linkname) == 0)
	break;
      if (errno != EEXIST)
	FAIL_EXIT1 ("symlink failed: %m");
      if (ntries++ == MAX_CREATE_LINK_TRIES)
	FAIL_EXIT1 ("symlink failed with EEXIST too many times");
    }
  add_temp_file (linkname);
}

#ifndef PATH_MAX
# define PATH_MAX 1024
#endif
static char valid_link[PATH_MAX];
static char dangling_link[PATH_MAX];
static char dangling_dir[PATH_MAX];

static void
do_prepare (int argc, char *argv[])
{
  char *fname;

  create_temp_file ("tst-glob_symlinks.", &fname);

  /* Create an existing symlink.  */
  create_link ("valid-symlink-tst-glob_symlinks", fname, valid_link,
	       sizeof valid_link);

  /* Create a dangling symlink to a file.  */
  int fd = create_temp_file ("dangling-tst-glob_file", &fname);
  TEST_VERIFY_EXIT (close (fd) == 0);
  /* It throws a warning at process end due 'add_temp_file' trying to
     unlink it again.  */
  TEST_VERIFY_EXIT (unlink (fname) == 0);
  create_link ("dangling-symlink-file-tst-glob", fname, dangling_link,
	       sizeof dangling_link);

  /* Create a dangling symlink to a directory.  */
  char tmpdir[PATH_MAX];
  snprintf (tmpdir, sizeof tmpdir, "%s/dangling-tst-glob_folder.XXXXXX",
	    test_dir);
  TEST_VERIFY_EXIT (mkdtemp (tmpdir) != NULL);
  create_link ("dangling-symlink-dir-tst-glob", tmpdir, dangling_dir,
	       sizeof dangling_dir);
  TEST_VERIFY_EXIT (rmdir (tmpdir) == 0);
}

static int
do_test (void)
{
  char buf[PATH_MAX + 1];
  glob_t gl;

  TEST_VERIFY_EXIT (glob (valid_link, 0, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], valid_link) == 0);
  globfree (&gl);

  TEST_VERIFY_EXIT (glob (dangling_link, 0, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], dangling_link) == 0);
  globfree (&gl);

  TEST_VERIFY_EXIT (glob (dangling_dir, 0, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], dangling_dir) == 0);
  globfree (&gl);

  snprintf (buf, sizeof buf, "%s", dangling_link);
  buf[strlen(buf) - 1] = '?';
  TEST_VERIFY_EXIT (glob (buf, 0, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], dangling_link) == 0);
  globfree (&gl);

  /* glob should handle dangling symbol as normal file, so <file>? should
     return an empty string.  */
  snprintf (buf, sizeof buf, "%s?", dangling_link);
  TEST_VERIFY_EXIT (glob (buf, 0, NULL, &gl) != 0);
  globfree (&gl);

  snprintf (buf, sizeof buf, "%s*", dangling_link);
  TEST_VERIFY_EXIT (glob (buf, 0, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], dangling_link) == 0);
  globfree (&gl);

  return 0;
}
