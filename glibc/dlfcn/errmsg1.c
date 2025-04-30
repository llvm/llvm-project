/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <string.h>

int
main (void)
{
  void *h;
  const char *s;

  /* Test that dlerror works initially.  */
  s = dlerror ();
  printf ("dlerror() without prior dl*() call returned: %s\n", s);
  if (s != NULL)
    return 1;

  h = dlopen ("errmsg1mod.so", RTLD_NOW);
  if (h != NULL)
    {
      dlclose (h);
      puts ("errmsg1mod.so could be loaded !?");
      exit (1);
    }

  s = dlerror ();
  puts (s);

  return strstr (s, "errmsg1mod.so") == NULL;
}
