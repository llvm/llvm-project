/* Test program for returning the canonical absolute name of a given file.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger <davidm@azstarnet.com>.

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

/* This file must be run from within a directory called "stdlib".  */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/stat.h>

/* Prototype for our test function.  */
extern int do_test (int argc, char *argv[]);
#include <test-skeleton.c>

#ifndef PATH_MAX
# define PATH_MAX 4096
#endif
static char	cwd[PATH_MAX];
static size_t	cwd_len;

struct {
  const char *	name;
  const char *	value;
} symlinks[] = {
  {"SYMLINK_LOOP",	"SYMLINK_LOOP"},
  {"SYMLINK_1",		"."},
  {"SYMLINK_2",		"//////./../../etc"},
  {"SYMLINK_3",		"SYMLINK_1"},
  {"SYMLINK_4",		"SYMLINK_2"},
  {"SYMLINK_5",		"doesNotExist"},
};

struct {
  const char * in, * out, * resolved;
  int error;
} tests[] = {
  /*  0 */
  {"/",					"/"},
  {"/////////////////////////////////",	"/"},
  {"/.././.././.././..///",		"/"},
  {"/etc",				"/etc"},
  {"/etc/../etc",			"/etc"},
  /*  5 */
  {"/doesNotExist/../etc",		0, "/doesNotExist", ENOENT},
  {"./././././././././.",		"."},
  {"/etc/.//doesNotExist",		0, "/etc/doesNotExist", ENOENT},
  {"./doesExist",			"./doesExist"},
  {"./doesExist/",			"./doesExist"},
  /* 10 */
  {"./doesExist/../doesExist",		"./doesExist"},
  {"foobar",				0, "./foobar", ENOENT},
  {".",					"."},
  {"./foobar",				0, "./foobar", ENOENT},
  {"SYMLINK_LOOP",			0, "./SYMLINK_LOOP", ELOOP},
  /* 15 */
  {"./SYMLINK_LOOP",			0, "./SYMLINK_LOOP", ELOOP},
  {"SYMLINK_1",				"."},
  {"SYMLINK_1/foobar",			0, "./foobar", ENOENT},
  {"SYMLINK_2",				"/etc"},
  {"SYMLINK_3",				"."},
  /* 20 */
  {"SYMLINK_4",				"/etc"},
  {"../stdlib/SYMLINK_1",		"."},
  {"../stdlib/SYMLINK_2",		"/etc"},
  {"../stdlib/SYMLINK_3",		"."},
  {"../stdlib/SYMLINK_4",		"/etc"},
  /* 25 */
  {"./SYMLINK_5",			0, "./doesNotExist", ENOENT},
  {"SYMLINK_5",				0, "./doesNotExist", ENOENT},
  {"SYMLINK_5/foobar",			0, "./doesNotExist", ENOENT},
  {"doesExist/../../stdlib/doesExist",	"./doesExist"},
  {"doesExist/.././../stdlib/.",	"."},
  /* 30 */
  {"./doesExist/someFile/",		0, "./doesExist/someFile", ENOTDIR},
  {"./doesExist/someFile/..",		0, "./doesExist/someFile", ENOTDIR},
};


static int
check_path (const char * result, const char * expected)
{
  int good;

  if (!result)
    return (expected == NULL);

  if (!expected)
    return 0;

  if (expected[0] == '.' && (expected[1] == '/' || expected[1] == '\0'))
    good = (strncmp (result, cwd, cwd_len) == 0
	    && strcmp (result + cwd_len, expected + 1) == 0);
  else
    good = (strcmp (expected, result) == 0);

  return good;
}


int
do_test (int argc, char ** argv)
{
  char * result;
  int i, errors = 0;
  char buf[PATH_MAX];

  getcwd (cwd, sizeof (buf));
  cwd_len = strlen (cwd);

  errno = 0;
  if (realpath (NULL, buf) != NULL || errno != EINVAL)
    {
      printf ("%s: expected return value NULL and errno set to EINVAL"
	      " for realpath(NULL,...)\n", argv[0]);
      ++errors;
    }

#if 0
  /* This is now allowed.  The test is invalid.  */
  errno = 0;
  if (realpath ("/", NULL) != NULL || errno != EINVAL)
    {
      printf ("%s: expected return value NULL and errno set to EINVAL"
	      " for realpath(...,NULL)\n", argv[0]);
      ++errors;
    }
#endif

  errno = 0;
  if (realpath ("", buf) != NULL || errno != ENOENT)
    {
      printf ("%s: expected return value NULL and set errno to ENOENT"
	      " for realpath(\"\",...)\n", argv[0]);
      ++errors;
    }

  for (i = 0; i < (int) (sizeof (symlinks) / sizeof (symlinks[0])); ++i)
    symlink (symlinks[i].value, symlinks[i].name);

  int has_dir = mkdir ("doesExist", 0777) == 0;

  int fd = has_dir ? creat ("doesExist/someFile", 0777) : -1;

  for (i = 0; i < (int) (sizeof (tests) / sizeof (tests[0])); ++i)
    {
      buf[0] = '\0';
      result = realpath (tests[i].in, buf);

      if (!check_path (result, tests[i].out))
	{
	  printf ("%s: flunked test %d (expected `%s', got `%s')\n",
		  argv[0], i, tests[i].out ? tests[i].out : "NULL",
		  result ? result : "NULL");
	  ++errors;
	  continue;
	}

      if (!check_path (buf, tests[i].out ? tests[i].out : tests[i].resolved))
	{
	  printf ("%s: flunked test %d (expected resolved `%s', got `%s')\n",
		  argv[0], i, tests[i].out ? tests[i].out : tests[i].resolved,
		  buf);
	  ++errors;
	  continue;
	}

      if (!tests[i].out && errno != tests[i].error)
	{
	  printf ("%s: flunked test %d (expected errno %d, got %d)\n",
		  argv[0], i, tests[i].error, errno);
	  ++errors;
	  continue;
	}

      char *result2 = realpath (tests[i].in, NULL);
      if ((result2 == NULL && result != NULL)
	  || (result2 != NULL && strcmp (result, result2) != 0))
	{
	  printf ("\
%s: realpath(..., NULL) produced different result than realpath(..., buf): '%s' vs '%s'\n",
		  argv[0], result2, result);
	  ++errors;
	}
      free (result2);
    }

  getcwd (buf, sizeof (buf));
  if (strcmp (buf, cwd))
    {
      printf ("%s: current working directory changed from %s to %s\n",
	      argv[0], cwd, buf);
      ++errors;
    }

  if (fd >= 0)
    {
      close (fd);
      unlink ("doesExist/someFile");
    }

  if (has_dir)
    rmdir ("doesExist");

  for (i = 0; i < (int) (sizeof (symlinks) / sizeof (symlinks[0])); ++i)
    unlink (symlinks[i].name);

  if (errors != 0)
    {
      printf ("%d errors.\n", errors);
      return EXIT_FAILURE;
    }

  puts ("No errors.");
  return EXIT_SUCCESS;
}
