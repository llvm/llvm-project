/* Test unrelated dlopen after dlopen failure involving NODELETE.
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
#include <errno.h>
#include <gnu/lib-names.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/xdlfcn.h>

static int
do_test (void)
{
  TEST_VERIFY (dlsym (NULL, "no_delete_mod_function") == NULL);

  /* This is expected to fail because of the missing dependency.  */
  puts ("info: attempting to load tst-dlopenfailmod1.so");
  TEST_VERIFY (dlopen ("tst-dlopenfailmod1.so", RTLD_LAZY) == NULL);
  const char *message = dlerror ();
  TEST_COMPARE_STRING (message,
                       "tst-dlopenfail-missingmod.so:"
                       " cannot open shared object file:"
                       " No such file or directory");

  /* Open a small shared object.  With a dangling GL (dl_initfirst)
     pointer, this is likely to crash because there is no longer any
     mapped text segment there (bug 25396).  */

  puts ("info: attempting to load tst-dlopenfailmod3.so");
  xdlclose (xdlopen ("tst-dlopenfailmod3.so", RTLD_NOW));

  return 0;
}

/* Do not perturb the dangling link map.  With M_PERTURB, the link map
   appears to have l_init_called set, so there are no constructor
   calls and no crashes.  */
#define TEST_NO_MALLOPT
#include <support/test-driver.c>
