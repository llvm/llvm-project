/* Check that a moved versioned symbol can be found using dlsym, dlvsym.
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

#include <stddef.h>
#include <support/check.h>
#include <support/xdlfcn.h>

static int
do_test (void)
{
  /* tst-sonamemove-runmod1.so does not define moved_function, but it
     depends on tst-sonamemove-runmod2.so, which does.  */
  void *handle = xdlopen ("tst-sonamemove-runmod1.so", RTLD_NOW);
  TEST_VERIFY (xdlsym (handle, "moved_function") != NULL);
  TEST_VERIFY (xdlvsym (handle, "moved_function", "SONAME_MOVE") != NULL);

  return 0;
}

#include <support/test-driver.c>
