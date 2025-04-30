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
#include <string.h>

#include <support/check.h>

static int
do_test (void)
{
  static const char modname[] = "tst-cet-legacy-mod-4.so";
  int (*fp) (void);
  void *h;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      const char *err = dlerror ();
      if (!strstr (err, "rebuild shared object with IBT support enabled"))
	FAIL_EXIT1 ("incorrect dlopen '%s' error: %s\n", modname, err);
      return 0;
    }

  fp = dlsym (h, "test");
  if (fp == NULL)
    FAIL_EXIT1 ("cannot get symbol 'test': %s\n", dlerror ());

  if (fp () != 0)
    FAIL_EXIT1 ("test () != 0");

  dlclose (h);

  return 0;
}

#include <support/test-driver.c>
