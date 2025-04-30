/* Unit test for dlopen on ELF object from "objcopy --only-keep-debug".
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

static int
do_test (void)
{
  void *h = dlopen ("tst-debug1mod1.so", RTLD_LAZY);
  if (h != NULL)
    {
      puts ("shouldn't load tst-debug1mod1.so");
      return 1;
    }
  return 0;
}

#include <support/test-driver.c>
