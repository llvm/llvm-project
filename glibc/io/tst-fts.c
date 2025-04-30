/* Simple test for some fts functions.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fts.h>

#include <errno.h>
#include <error.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void prepare (void);
static int do_test (void);
#define PREPARE(argc, argv)     prepare ()
#define TEST_FUNCTION           do_test ()
#include "../test-skeleton.c"

static char *fts_test_dir;

static void
make_dir (const char *dirname)
{
  char *name;
  if (asprintf (&name, "%s/%s", fts_test_dir, dirname) < 0)
    {
      puts ("out of memory");
      exit (1);
    }

  if (mkdir (name, 0700) < 0)
    {
      printf ("cannot create dir \"%s\": %m\n", name);
      exit (1);
    }

  add_temp_file (name);
}

static void
make_file (const char *filename)
{
  char *name;
  if (asprintf (&name, "%s/%s", fts_test_dir, filename) < 0)
    {
      puts ("out of memory");
      exit (1);
    }

  int fd = open (name, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0)
    {
      printf ("cannot create file \"%s\": %m\n", name);
      exit (1);
    }
  close (fd);

  add_temp_file (name);
}

static void
prepare (void)
{
  char *dirbuf;
  char dir_name[] = "/tst-fts.XXXXXX";

  if (asprintf (&dirbuf, "%s%s", test_dir, dir_name) < 0)
    {
      puts ("out of memory");
      exit (1);
    }

  if (mkdtemp (dirbuf) == NULL)
    {
      puts ("cannot create temporary directory");
      exit (1);
    }

  add_temp_file (dirbuf);
  fts_test_dir = dirbuf;

  make_file ("12");
  make_file ("345");
  make_file ("6789");

  make_dir ("aaa");
  make_file ("aaa/1234");
  make_file ("aaa/5678");

  make_dir ("bbb");
  make_file ("bbb/1234");
  make_file ("bbb/5678");
  make_file ("bbb/90ab");
}

/* Largest name wins, otherwise strcmp.  */
static int
compare_ents (const FTSENT **ent1, const FTSENT **ent2)
{
  short len1 = (*ent1)->fts_namelen;
  short len2 = (*ent2)->fts_namelen;
  if (len1 != len2)
    return len1 - len2;
  else
    {
      const char *name1 = (*ent1)->fts_name;
      const char *name2 = (*ent2)->fts_name;
      return strcmp (name1, name2);
    }
}

/* Count the number of files seen as children.  */
static int files = 0;

static void
children (FTS *fts)
{
  FTSENT *child = fts_children (fts, 0);
  if (child == NULL && errno != 0)
    {
      printf ("FAIL: fts_children: %m\n");
      exit (1);
    }

  while (child != NULL)
    {
      short level = child->fts_level;
      const char *name = child->fts_name;
      if (child->fts_info == FTS_F || child->fts_info == FTS_NSOK)
	{
	  files++;
	  printf ("%*s%s\n", 2 * level, "", name);
	}
      child = child->fts_link;
    }
}

/* Count the number of dirs seen in the test.  */
static int dirs = 0;

static int
do_test (void)
{
  char *paths[2] = { fts_test_dir, NULL };
  FTS *fts;
  fts = fts_open (paths, FTS_LOGICAL, &compare_ents);
  if (fts == NULL)
    {
      printf ("FAIL: fts_open: %m\n");
      exit (1);
    }

  FTSENT *ent;
  while ((ent = fts_read (fts)) != NULL)
    {
      const char *name = ent->fts_name;
      short level = ent->fts_level;
      switch (ent->fts_info)
	{
	case FTS_F:
	  /* Don't show anything, children will have on parent dir.  */
	  break;

	case FTS_D:
	  printf ("%*s%s =>\n", 2 * level, "", name);
	  children (fts);
	  break;

	case FTS_DP:
	  dirs++;
	  printf ("%*s<= %s\n", 2 * level, "", name);
	  break;

	case FTS_NS:
	case FTS_ERR:
	  printf ("FAIL: fts_read ent: %s\n", strerror (ent->fts_errno));
	  exit (1);
	  break;

	default:
	  printf ("FAIL: unexpected fts_read ent %s\n", name);
	  exit (1);
	  break;
	}
    }
  /* fts_read returns NULL when done (and clears errno)
     or when an error occured (with errno set).  */
  if (errno != 0)
    {
      printf ("FAIL: fts_read: %m\n");
      exit (1);
    }

  if (fts_close (fts) != 0)
    {
      printf ("FAIL: fts_close: %m\n");
      exit (1);
    }

  if (files != 8)
    {
      printf ("FAIL: Unexpected number of files: %d\n", files);
      return 1;
    }

  if (dirs != 3)
    {
      printf ("FAIL: Unexpected number of dirs: %d\n", dirs);
      return 1;
    }

  puts ("PASS");
  return 0;
}
