/* Helper module to load tst-dlopen-nodelete-reloc-mod9.so.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

static void *handle;

static void __attribute__ ((constructor))
init (void)
{
  handle = dlopen ("tst-dlopen-nodelete-reloc-mod9.so", RTLD_NOW);
  if (handle == NULL)
    {
      printf ("error: dlopen in module 8: %s\n", dlerror ());
      _exit (1);
    }
}

static void __attribute__ ((destructor))
fini (void)
{
  dlclose (handle);
}
