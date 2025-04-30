/* Global-scope DSO mapping test with a static executable (BZ #15022).
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <stddef.h>
#include <stdio.h>

#define MAGIC0 0
#define MAGIC1 0x5500ffaa
#define MAGIC2 0xaaff0055

/* Mapping a DSO into the global scope used to crash in static
   executables.  Check that it succeeds and then that symbols from
   the DSO can be accessed and operate as expected.  */
static int
do_test (void)
{
  unsigned int (*getfoo) (void);
  void (*setfoo) (unsigned int);
  unsigned int *foop;
  unsigned int foo;
  void *handle;

  /* Try to map a module into the global scope.  */
  handle = dlopen ("modstatic3.so", RTLD_LAZY | RTLD_GLOBAL);
  if (handle == NULL)
    {
      printf ("dlopen (modstatic3.so): %s\n", dlerror ());
      return 1;
    }

  /* Get at its symbols.  */
  foop = dlsym (handle, "foo");
  if (foop == NULL)
    {
      printf ("dlsym (foo): %s\n", dlerror ());
      return 1;
    }

  getfoo = dlsym (handle, "getfoo");
  if (getfoo == NULL)
    {
      printf ("dlsym (getfoo): %s\n", dlerror ());
      return 1;
    }

  setfoo = dlsym (handle, "setfoo");
  if (setfoo == NULL)
    {
      printf ("dlsym (setfoo): %s\n", dlerror ());
      return 1;
    }

  /* Make sure the view of the initial state is consistent.  */
  foo = *foop;
  if (foo != MAGIC0)
    {
      printf ("*foop: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  foo = getfoo ();
  if (foo != MAGIC0)
    {
      printf ("getfoo: got %#x, expected %#x\n", foo, MAGIC0);
      return 1;
    }

  /* Likewise with one change to its state.  */
  setfoo (MAGIC1);

  foo = *foop;
  if (foo != MAGIC1)
    {
      printf ("*foop: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  foo = getfoo ();
  if (foo != MAGIC1)
    {
      printf ("getfoo: got %#x, expected %#x\n", foo, MAGIC1);
      return 1;
    }

  /* And with another.  */
  setfoo (MAGIC2);

  foo = *foop;
  if (foo != MAGIC2)
    {
      printf ("*foop: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  foo = getfoo ();
  if (foo != MAGIC2)
    {
      printf ("getfoo: got %#x, expected %#x\n", foo, MAGIC2);
      return 1;
    }

  /* All done, clean up.  */
  getfoo = NULL;
  setfoo = NULL;
  foop = NULL;
  dlclose (handle);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
