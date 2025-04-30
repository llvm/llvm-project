/* Check compatibility of CET-enabled executable with dlopened legacy
   shared object.
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
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <x86intrin.h>
#include <support/check.h>

#if defined CET_IS_PERMISSIVE || defined CET_DISABLED_BY_ENV
# define CET_MAYBE_DISABLED 1
#else
# define CET_MAYBE_DISABLED 0
#endif

static void
do_test_1 (const char *modname, bool fail)
{
  int (*fp) (void);
  void *h;

  /* NB: dlopen should never fail on non-CET platforms.  If SHSTK is
     disabled, assuming IBT is also disabled.  */
  bool cet_enabled = _get_ssp () != 0 && !CET_MAYBE_DISABLED;
  if (!cet_enabled)
    fail = false;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      const char *err = dlerror ();
      if (fail)
	{
	  if (strstr (err, "rebuild shared object with SHSTK support enabled")
	      == NULL)
	    FAIL_EXIT1 ("incorrect dlopen '%s' error: %s\n", modname, err);

	  return;
	}

      FAIL_EXIT1 ("cannot open '%s': %s\n", modname, err);
    }

  if (fail)
    FAIL_EXIT1 ("dlopen should have failed\n");

  fp = dlsym (h, "test");
  if (fp == NULL)
    {
      printf ("cannot get symbol 'test': %s\n", dlerror ());
      exit (1);
    }

  if (fp () != 0)
    {
      puts ("test () != 0");
      exit (1);
    }

  dlclose (h);
}

static int
do_test (void)
{
  do_test_1 ("tst-cet-legacy-mod-6a.so", true);
  do_test_1 ("tst-cet-legacy-mod-6b.so", false);
  return 0;
}

#include <support/test-driver.c>
