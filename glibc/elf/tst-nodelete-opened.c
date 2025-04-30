/* Verify that an already opened DSO opened agained with RTLD_NODELETE actually
   sets the NODELETE flag.

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

#include <dlfcn.h>
#include <stdio.h>

int
do_test (void)
{
  void *h1 = dlopen ("$ORIGIN/tst-nodelete-opened-lib.so", RTLD_LAZY);
  if (h1 == NULL)
    {
      printf ("h1: failed to open DSO: %s\n", dlerror ());
      return 1;
    }

  void *h2 = dlopen ("$ORIGIN/tst-nodelete-opened-lib.so",
		     RTLD_LAZY | RTLD_NODELETE);
  if (h2 == NULL)
    {
      printf ("h2: failed to open DSO: %s\n", dlerror ());
      return 1;
    }

  int *foo = dlsym (h2, "foo_var");
  if (foo == NULL)
    {
      printf ("failed to load symbol foo_var: %s\n", dlerror ());
      return 1;
    }

  if (dlclose (h1) != 0)
    {
      printf ("h1: dlclose failed: %s\n", dlerror ());
      return 1;
    }

  if (dlclose (h2) != 0)
    {
      printf ("h2: dlclose failed: %s\n", dlerror ());
      return 1;
    }

  /* This FOO dereference will crash with a segfault if the DSO was
     unloaded.  */
  printf ("foo == %d\n", *foo);

  return 0;
}

#include <support/test-driver.c>
