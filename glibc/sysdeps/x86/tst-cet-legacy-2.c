/* Check compatibility of CET-enabled executable with dlopened legacy
   shared object.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

extern int in_dso_2 (void);

static int
do_test (void)
{
  static const char modname[] = "tst-cet-legacy-mod-1.so";
  int (*fp) (void);
  void *h;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot open '%s': %s\n", modname, dlerror ());
      exit (1);
    }

  fp = dlsym (h, "in_dso_1");
  if (fp == NULL)
    {
      printf ("cannot get symbol 'in_dso': %s\n", dlerror ());
      exit (1);
    }

  if (fp () != 0x1234678)
    {
      puts ("in_dso_1 () != 0x1234678");
      exit (1);
    }

  if (in_dso_2 () != 0xbadbeef)
    {
      puts ("in_dso_2 () != 0xbadbeef");
      exit (1);
    }

  dlclose (h);

  return 0;
}

#include <support/test-driver.c>
