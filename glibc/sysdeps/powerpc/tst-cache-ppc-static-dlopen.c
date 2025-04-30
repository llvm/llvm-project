/* Test dl_cache_line_size from a dlopen'ed DSO from a static executable.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <errno.h>

int test_cache(int *);

static int
do_test (void)
{
  int ret;
  void *handle;
  int (*test_cache) (int *);

  handle = dlopen ("mod-cache-ppc.so", RTLD_LAZY | RTLD_LOCAL);
  if (handle == NULL)
    {
      printf ("dlopen (mod-cache-ppc.so): %s\n", dlerror ());
      return 1;
    }

  test_cache = dlsym (handle, "test_cache");
  if (test_cache == NULL)
    {
      printf ("dlsym (test_cache): %s\n", dlerror ());
      return 1;
    }

  ret = test_cache(&errno);

  test_cache = NULL;
  dlclose (handle);

  return ret;
}

#include <support/test-driver.c>
