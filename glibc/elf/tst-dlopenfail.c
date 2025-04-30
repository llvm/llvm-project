/* Test dlopen rollback after failures involving NODELETE objects (bug 20839).
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

  /* Do not probe for the presence of the tst-dlopenfailnodelmod.so at
     this point because that might trigger relocation if bug 20839 is
     present, obscuring a subsequent crash.  */

  /* This is expected to succeed.  */
  puts ("info: loading tst-dlopenfailmod2.so");
  void *handle = xdlopen ("tst-dlopenfailmod2.so", RTLD_NOW);
  xdlclose (handle);

  /* The NODELETE module should remain loaded.  */
  TEST_VERIFY (dlopen ("tst-dlopenfailnodelmod.so", RTLD_LAZY | RTLD_NOLOAD)
               != NULL);
  /* But the symbol is not in the global scope.  */
  TEST_VERIFY (dlsym (NULL, "no_delete_mod_function") == NULL);

  /* We can make tst-dlopenfailnodelmod.so global, and then the symbol
     should become available.  */
  TEST_VERIFY (dlopen ("tst-dlopenfailnodelmod.so", RTLD_LAZY | RTLD_GLOBAL)
               != NULL);
  void (*no_delete_mod_function) (void)
    = dlsym (NULL, "no_delete_mod_function");
  TEST_VERIFY_EXIT (no_delete_mod_function != NULL);

  /* Hopefully, no_delete_mod_function is sufficiently complex to
     depend on relocations.  */
  no_delete_mod_function ();

  return 0;
}

#include <support/test-driver.c>
