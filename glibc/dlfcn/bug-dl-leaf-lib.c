/* Make sure dlopen/dlclose are not marked as leaf functions.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mike Frysinger <vapier@gentoo.org>

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

/* The bug-dl-leaf.c file will call our lib_main directly.  We do this to
   keep things simple -- no need to use --export-dynamic with the linker
   or build the main ELF as a PIE.

   The lib_main func will modify some of its state while dlopening and
   dlclosing the bug-dl-leaf-lib-cb.so library.  The constructors and
   destructors in that library will call back into this library to also
   muck with state (the check_val_xxx funcs).

   If dlclose/dlopen are marked as "leaf" functions, then with newer
   versions of gcc, the state modification won't work correctly.  */

#include <assert.h>
#include <dlfcn.h>

static int val = 1;
static int called = 0;

void check_val_init (void)
{
  called = 1;
  assert (val == 2);
}

void check_val_fini (void)
{
  called = 2;
  assert (val == 4);
}

int lib_main (void)
{
  int ret __attribute__ ((unused));
  void *hdl;

  /* Make sure the constructor sees the updated val.  */
  val = 2;
  hdl = dlopen ("bug-dl-leaf-lib-cb.so", RTLD_GLOBAL | RTLD_LAZY);
  val = 3;
  assert (hdl);
  assert (called == 1);

  /* Make sure the destructor sees the updated val.  */
  val = 4;
  ret = dlclose (hdl);
  val = 5;
  assert (ret == 0);
  assert (called == 2);

  return !val;
}
