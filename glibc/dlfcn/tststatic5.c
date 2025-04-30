/* GLRO(dl_pagesize) initialization DSO test with a static executable.
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
#include <unistd.h>

/* Check that the same page size is reported both directly and by a DSO
   mapped from a static executable.

   On targets that support different page sizes, the kernel communicates
   the size currently in use via the auxiliary vector.  The auxiliary
   vector and HWCAP/HWCAP2 bits are copied across the static dlopen
   boundary in __rtld_static_init.  */
static int
do_test (void)
{
  int pagesize = getpagesize ();
  int (*my_getpagesize) (void);
  int my_pagesize;
  void *handle;

  /* Try to map a module.  */
  handle = dlopen ("modstatic5.so", RTLD_LAZY | RTLD_LOCAL);
  if (handle == NULL)
    {
      printf ("dlopen (modstatic5.so): %s\n", dlerror ());
      return 1;
    }

  /* Get at its symbol.  */
  my_getpagesize = dlsym (handle, "my_getpagesize");
  if (my_getpagesize == NULL)
    {
      printf ("dlsym (my_getpagesize): %s\n", dlerror ());
      return 1;
    }

  /* Make sure the page size reported is the same either way.  */
  my_pagesize = my_getpagesize ();
  if (my_pagesize != pagesize)
    {
      printf ("my_getpagesize: got %i, expected %i\n", my_pagesize, pagesize);
      return 1;
    }

  /* All done, clean up.  */
  my_getpagesize = NULL;
  dlclose (handle);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
