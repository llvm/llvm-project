/* Test glob memory management.
   for the filesystem access functions.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <error.h>
#include <dirent.h>
#include <glob.h>
#include <mcheck.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

// #define DEBUG
#ifdef DEBUG
# define PRINTF(fmt, args...) \
  do					\
    {					\
      int save_errno = errno;		\
      printf (fmt, ##args);		\
      errno = save_errno;		\
    } while (0)
#else
# define PRINTF(fmt, args...)
#endif

#define LONG_NAME \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

static struct
{
  const char *name;
  int level;
  int type;
  mode_t mode;
} filesystem[] =
{
  { ".", 1, DT_DIR, 0755 },
  { "..", 1, DT_DIR, 0755 },
  { "dir", 1, DT_DIR, 0755 },
    { ".", 2, DT_DIR, 0755 },
    { "..", 2, DT_DIR, 0755 },
    { "readable", 2, DT_DIR, 0755 },
      { ".", 3, DT_DIR, 0755 },
      { "..", 3, DT_DIR, 0755 },
      { "a", 3, DT_REG, 0644 },
      { LONG_NAME, 3, DT_REG, 0644 },
    { "unreadable", 2, DT_DIR, 0111 },
      { ".", 3, DT_DIR, 0111 },
      { "..", 3, DT_DIR, 0755 },
      { "a", 3, DT_REG, 0644 },
    { "zz-readable", 2, DT_DIR, 0755 },
      { ".", 3, DT_DIR, 0755 },
      { "..", 3, DT_DIR, 0755 },
      { "a", 3, DT_REG, 0644 }
};
#define nfiles (sizeof (filesystem) / sizeof (filesystem[0]))


typedef struct
{
  int level;
  int idx;
  struct dirent d;
  char room_for_dirent[sizeof (LONG_NAME)];
} my_DIR;


static long int
find_file (const char *s)
{
  int level = 1;
  long int idx = 0;

  if (strcmp (s, ".") == 0)
    return 0;

  if (s[0] == '.' && s[1] == '/')
    s += 2;

  while (*s != '\0')
    {
      char *endp = strchrnul (s, '/');

      PRINTF ("looking for %.*s, level %d\n", (int) (endp - s), s, level);

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
  my_DIR *dir;

  if (idx == -1)
    {
      PRINTF ("my_opendir(\"%s\") == NULL (%m)\n", s);
      return NULL;
    }

  if ((filesystem[idx].mode & 0400) == 0)
    {
      errno = EACCES;
      PRINTF ("my_opendir(\"%s\") == NULL (%m)\n", s);
      return NULL;
    }

  dir = (my_DIR *) malloc (sizeof (my_DIR));
  if (dir == NULL)
    {
      printf ("cannot allocate directory handle: %m\n");
      exit (EXIT_FAILURE);
    }

  dir->level = filesystem[idx].level;
  dir->idx = idx;

  PRINTF ("my_opendir(\"%s\") == { level: %d, idx: %ld }\n",
	  s, filesystem[idx].level, idx);

  return dir;
}


static struct dirent *
my_readdir (void *gdir)
{
  my_DIR *dir = gdir;

  if (dir->idx == -1)
    {
      PRINTF ("my_readdir ({ level: %d, idx: %ld }) = NULL\n",
	      dir->level, (long int) dir->idx);
      return NULL;
    }

  while (dir->idx < nfiles && filesystem[dir->idx].level > dir->level)
    ++dir->idx;

  if (dir->idx == nfiles || filesystem[dir->idx].level < dir->level)
    {
      dir->idx = -1;
      PRINTF ("my_readdir ({ level: %d, idx: %ld }) = NULL\n",
	      dir->level, (long int) dir->idx);
      return NULL;
    }

  dir->d.d_ino = 1;		/* glob should not skip this entry.  */

  dir->d.d_type = filesystem[dir->idx].type;

  strcpy (dir->d.d_name, filesystem[dir->idx].name);

  PRINTF ("my_readdir ({ level: %d, idx: %ld }) = { d_ino: %ld, d_type: %d, d_name: \"%s\" }\n",
	  dir->level, (long int) dir->idx, dir->d.d_ino, dir->d.d_type,
	  dir->d.d_name);

  ++dir->idx;

  return &dir->d;
}


static void
my_closedir (void *dir)
{
  PRINTF ("my_closedir ()\n");
  free (dir);
}


/* We use this function for lstat as well since we don't have any.  */
static int
my_stat (const char *name, struct stat *st)
{
  long int idx = find_file (name);

  if (idx == -1)
    {
      PRINTF ("my_stat (\"%s\", ...) = -1 (%m)\n", name);
      return -1;
    }

  memset (st, '\0', sizeof (*st));

  if (filesystem[idx].type == DT_UNKNOWN)
    st->st_mode = DTTOIF (idx + 1 < nfiles
			  && filesystem[idx].level < filesystem[idx + 1].level
			  ? DT_DIR : DT_REG) | filesystem[idx].mode;
  else
    st->st_mode = DTTOIF (filesystem[idx].type) | filesystem[idx].mode;

  PRINTF ("my_stat (\"%s\", { st_mode: %o }) = 0\n", name, st->st_mode);

  return 0;
}


static void
init_glob_altdirfuncs (glob_t *pglob)
{
  pglob->gl_closedir = my_closedir;
  pglob->gl_readdir = my_readdir;
  pglob->gl_opendir = my_opendir;
  pglob->gl_lstat = my_stat;
  pglob->gl_stat = my_stat;
}


int
do_test (void)
{
  mtrace ();

  glob_t gl;
  memset (&gl, 0, sizeof (gl));
  init_glob_altdirfuncs (&gl);

  if (glob ("dir/*able/*", GLOB_ERR | GLOB_ALTDIRFUNC, NULL, &gl)
      != GLOB_ABORTED)
    {
      puts ("glob did not fail with GLOB_ABORTED");
      exit (EXIT_FAILURE);
    }

  globfree (&gl);

  memset (&gl, 0, sizeof (gl));
  init_glob_altdirfuncs (&gl);

  gl.gl_offs = 3;
  if (glob ("dir2/*", GLOB_DOOFFS, NULL, &gl) != GLOB_NOMATCH)
    {
      puts ("glob did not fail with GLOB_NOMATCH");
      exit (EXIT_FAILURE);
    }

  globfree (&gl);

  muntrace ();

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
