/* Test for dependency tracking  added by relocations.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>


static void *
load (const char *name)
{
  void *d = dlopen (name, RTLD_LAZY | RTLD_GLOBAL);
  if (d == NULL)
    error (EXIT_FAILURE, errno, "cannot load `%s'", name);
  return d;
}


#define TEST_FUNCTION do_test ()
extern int do_test (void);

int
do_test (void)
{
  void *d1;
  void *d2;
  int (*f) (void);

  d1 = load ("glreflib1.so");
  d2 = load ("glreflib2.so");

  f = dlsym (d2, "ref2");
  if (f == NULL)
    error (EXIT_FAILURE, errno, "cannot get pointer to `%s'", "ref2");

  if (f () != 42)
    error (EXIT_FAILURE, 0, "wrong result from `%s'", "ref2");

  puts ("Correct result in first call");
  fflush (stdout);

  /* Now unload the first file.  */
  dlclose (d1);

  puts ("About to call the second time");
  fflush (stdout);

  /* Try calling the function again.  */
  if (f () != 42)
    error (EXIT_FAILURE, 0, "wrong result from `%s' (second call)", "ref2");

  puts ("Second call succeeded!");
  fflush (stdout);

  dlclose (d2);

  puts ("glreflib2 also closed");
  fflush (stdout);

  return 0;
}

#include "../test-skeleton.c"
