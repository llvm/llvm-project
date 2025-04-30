/* Verify that RTLD_NOLOAD works as expected.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <gnu/lib-names.h>

static int
do_test (void)
{
  /* Test that no object is loaded with RTLD_NOLOAD.  */
  void *h1 = dlopen (LIBM_SO, RTLD_LAZY | RTLD_NOLOAD);
  if (h1 != NULL)
    {
      printf ("h1: DSO has been loaded while it should have not\n");
      return 1;
    }

  /* This used to segfault in some glibc versions.  */
  void *h2 = dlopen (LIBM_SO, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
  if (h2 != NULL)
    {
      printf ("h2: DSO has been loaded while it should have not\n");
      return 1;
    }

  /* Test that loading an already loaded object returns the same.  */
  void *h3 = dlopen (LIBM_SO, RTLD_LAZY);
  if (h3 == NULL)
    {
      printf ("h3: failed to open DSO: %s\n", dlerror ());
      return 1;
    }
  void *h4 = dlopen (LIBM_SO, RTLD_LAZY | RTLD_NOLOAD);
  if (h4 == NULL)
    {
      printf ("h4: failed to open DSO: %s\n", dlerror ());
      return 1;
    }
  if (h4 != h3)
    {
      printf ("h4: should return the same object\n");
      return 1;
    }

  /* Cleanup */
  if (dlclose (h3) != 0)
    {
      printf ("h3: dlclose failed: %s\n", dlerror ());
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
