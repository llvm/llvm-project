/* Test for realpath/canonicalize function.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <string.h>


/* Prototype for our test function.  */
extern void do_prepare (int argc, char *argv[]);
extern int do_test (int argc, char *argv[]);

/* We have a preparation function.  */
#define PREPARE do_prepare

#include <test-skeleton.c>

/* Name of the temporary files we create.  */
char *name1;
char *name2;

/* Preparation.  */
void
do_prepare (int argc, char *argv[])
{
  size_t test_dir_len;

  test_dir_len = strlen (test_dir);

  /* Generate the circular symlinks.  */
  name1 = malloc (test_dir_len + sizeof ("/canonXXXXXX"));
  mempcpy (mempcpy (name1, test_dir, test_dir_len),
	   "/canonXXXXXX", sizeof ("/canonXXXXXX"));
  name2 = strdup (name1);

  add_temp_file (mktemp (name1));
  add_temp_file (mktemp (name2));
}


/* Run the test.  */
int
do_test (int argc, char *argv[])
{
  char *canon;

  printf ("create symlinks from %s to %s and vice versa\n", name1, name2);
  if (symlink (name1, name2) == -1
      || symlink (name2, name1) == -1)
    /* We cannot test this.  */
    return 0;

  /* Call the function.  This is equivalent the using `realpath' but the
     function allocates the room for the result.  */
  errno = 0;
  canon = canonicalize_file_name (name1);

  return canon != NULL || errno != ELOOP;
}
