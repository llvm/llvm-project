/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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


int
main (void)
{
  const char fname[] = "modatexit.so";
  void *h;
  void (*fp) (void *);
  int v = 0;

  h = dlopen (fname, RTLD_NOW);
  if (h == NULL)
    {
      printf ("cannot open \"%s\": %s\n", fname, dlerror ());
      exit (1);
    }

  fp = dlsym (h, "foo");
  if (fp == NULL)
    {
      printf ("cannot find \"foo\": %s\n", dlerror ());
      exit (1);
    }

  fp (&v);

  if (dlclose (h) != 0)
    {
      printf ("cannot close \"%s\": %s\n", fname, dlerror ());
      exit (1);
    }

  if (v != 1)
    {
      puts ("module unload didn't change `v'");
      exit (1);
    }

  puts ("finishing now");

  return 0;
}
