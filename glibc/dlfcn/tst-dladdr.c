/* Test for dladdr.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Volkmar Sieh <vs@caldera.de> and Andreas Jaeger <aj@suse.de>.

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
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_FUNCTION do_test ()
extern int do_test (void);

int
do_test (void)
{
  void *handle;
  int (*sym) (void); /* We load ref1 from glreflib1.c.  */
  Dl_info info;
  int ret;


  handle = dlopen ("glreflib1.so", RTLD_NOW);
  if (handle == NULL)
    error (EXIT_FAILURE, 0, "cannot load: glreflib1.so");

  sym = dlsym (handle, "ref1");
  if (sym == NULL)
    error (EXIT_FAILURE, 0, "dlsym failed");

  memset (&info, 0, sizeof (info));
  ret = dladdr (sym, &info);

  if (ret == 0)
    error (EXIT_FAILURE, 0, "dladdr failed");

  printf ("ret = %d\n", ret);
  printf ("info.dli_fname = %p (\"%s\")\n", info.dli_fname, info.dli_fname);
  printf ("info.dli_fbase = %p\n", info.dli_fbase);
  printf ("info.dli_sname = %p (\"%s\")\n", info.dli_sname, info.dli_sname);
  printf ("info.dli_saddr = %p\n", info.dli_saddr);

  if (info.dli_fname == NULL)
    error (EXIT_FAILURE, 0, "dli_fname is NULL");
  if (info.dli_fbase == NULL)
    error (EXIT_FAILURE, 0, "dli_fbase is NULL");
  if (info.dli_sname == NULL)
    error (EXIT_FAILURE, 0, "dli_sname is NULL");
  if (info.dli_saddr == NULL)
    error (EXIT_FAILURE, 0, "dli_saddr is NULL");

  dlclose (handle);

  return 0;
}


#include "../test-skeleton.c"
