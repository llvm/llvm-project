/* Test glob compat symbol which avoid call GLOB_ALTDIRFUNC/gl_lstat.
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

#include <glob.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>

#include <shlib-compat.h>
#include <support/check.h>
#include <support/temp_file.h>

__typeof (glob) glob;
/* On alpha glob exists in version GLIBC_2_0, GLIBC_2_1, and GLIBC_2_27.
   This test needs to access the version prior to GLIBC_2_27, which is
   GLIBC_2_1 on alpha, GLIBC_2_0 elsewhere.  */
#ifdef __alpha__
compat_symbol_reference (libc, glob, glob, GLIBC_2_1);
#else
compat_symbol_reference (libc, glob, glob, GLIBC_2_0);
#endif

/* Compat glob should not call gl_lstat since for some old binaries it
   might be unitialized (for instance GNUmake).  Check if it is indeed
   not called.  */
static bool stat_called;
static bool lstat_called;

static struct
{
  const char *name;
  int level;
  int type;
} filesystem[] =
{
  { ".", 1, DT_DIR },
  { "..", 1, DT_DIR },
  { "dir1lev1", 1, DT_UNKNOWN },
    { ".", 2, DT_DIR },
    { "..", 2, DT_DIR },
    { "file1lev2", 2, DT_REG },
    { "file2lev2", 2, DT_REG },
};
static const size_t nfiles = sizeof (filesystem) / sizeof (filesystem [0]);

typedef struct
{
  int level;
  int idx;
  struct dirent d;
  char room_for_dirent[NAME_MAX];
} my_DIR;

static long int
find_file (const char *s)
{
  int level = 1;
  long int idx = 0;

  while (s[0] == '/')
    {
      if (s[1] == '\0')
	{
	  s = ".";
	  break;
	}
      ++s;
    }

  if (strcmp (s, ".") == 0)
    return 0;

  if (s[0] == '.' && s[1] == '/')
    s += 2;

  while (*s != '\0')
    {
      char *endp = strchrnul (s, '/');

      while (idx < nfiles && filesystem[idx].level >= level)
	{
	  if (filesystem[idx].level == level
	      && memcmp (s, filesystem[idx].name, endp - s) == 0
	      && filesystem[idx].name[endp - s] == '\0')
	    break;
	  ++idx;
	}

      if (idx == nfiles || filesystem[idx].level < level)
	{
	  errno = ENOENT;
	  return -1;
	}

      if (*endp == '\0')
	return idx + 1;

      if (filesystem[idx].type != DT_DIR
	  && (idx + 1 >= nfiles
	      || filesystem[idx].level >= filesystem[idx + 1].level))
	{
	  errno = ENOTDIR;
	  return -1;
	}

      ++idx;

      s = endp + 1;
      ++level;
    }

  errno = ENOENT;
  return -1;
}

static void *
my_opendir (const char *s)
{
  long int idx = find_file (s);
  if (idx == -1 || filesystem[idx].type != DT_DIR)
    return NULL;

  my_DIR *dir = malloc (sizeof (my_DIR));
  if (dir == NULL)
    FAIL_EXIT1 ("cannot allocate directory handle");

  dir->level = filesystem[idx].level;
  dir->idx = idx;

  return dir;
}

static struct dirent *
my_readdir (void *gdir)
{
  my_DIR *dir = gdir;

  if (dir->idx == -1)
    return NULL;

  while (dir->idx < nfiles && filesystem[dir->idx].level > dir->level)
    ++dir->idx;

  if (dir->idx == nfiles || filesystem[dir->idx].level < dir->level)
    {
      dir->idx = -1;
      return NULL;
    }

  dir->d.d_ino = 1;		/* glob should not skip this entry.  */

  dir->d.d_type = filesystem[dir->idx].type;

  strcpy (dir->d.d_name, filesystem[dir->idx].name);

  ++dir->idx;

  return &dir->d;
}

static void
my_closedir (void *dir)
{
  free (dir);
}

static int
my_stat (const char *name, struct stat *st)
{
  stat_called = true;

  long int idx = find_file (name);
  if (idx == -1)
    return -1;

  memset (st, '\0', sizeof (*st));

  if (filesystem[idx].type == DT_UNKNOWN)
    st->st_mode = DTTOIF (idx + 1 < nfiles
			  && filesystem[idx].level < filesystem[idx + 1].level
			  ? DT_DIR : DT_REG) | 0777;
  else
    st->st_mode = DTTOIF (filesystem[idx].type) | 0777;
  return 0;
}

static int
my_lstat (const char *name, struct stat *st)
{
  lstat_called = true;

  long int idx = find_file (name);
  if (idx == -1)
    return -1;

  memset (st, '\0', sizeof (*st));

  if (filesystem[idx].type == DT_UNKNOWN)
    st->st_mode = DTTOIF (idx + 1 < nfiles
			  && filesystem[idx].level < filesystem[idx + 1].level
			  ? DT_DIR : DT_REG) | 0777;
  else
    st->st_mode = DTTOIF (filesystem[idx].type) | 0777;
  return 0;
}

static int
do_test (void)
{
  glob_t gl;

  memset (&gl, '\0', sizeof (gl));

  gl.gl_closedir = my_closedir;
  gl.gl_readdir = my_readdir;
  gl.gl_opendir = my_opendir;
  gl.gl_lstat = my_lstat;
  gl.gl_stat = my_stat;

  int flags = GLOB_ALTDIRFUNC;

  stat_called = false;
  lstat_called = false;

  TEST_VERIFY_EXIT (glob ("*/file1lev2", flags, NULL, &gl) == 0);
  TEST_VERIFY_EXIT (gl.gl_pathc == 1);
  TEST_VERIFY_EXIT (strcmp (gl.gl_pathv[0], "dir1lev1/file1lev2") == 0);

  TEST_VERIFY_EXIT (stat_called == true);
  TEST_VERIFY_EXIT (lstat_called == false);

  return 0;
}

#include <support/test-driver.c>
