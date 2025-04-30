/* Verify that TLS access in separate thread in a dlopened library does not
   deadlock.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* When one dynamically loads a module, which spawns a thread to perform some
   activities, it could be possible that TLS storage is accessed for the first
   time in that thread.  This results in an allocation request within the
   thread, which could result in an attempt to take the rtld load_lock.  This
   is a problem because it would then deadlock with the dlopen (which owns the
   lock), if the main thread is waiting for the spawned thread to exit.  We can
   at least ensure that this problem does not occur due to accesses within
   libc.so, by marking TLS variables within libc.so as IE.  The problem of an
   arbitrary variable being accessed and constructed within such a thread still
   exists but this test case does not verify that.  */

int
do_test (void)
{
  void *f = dlopen ("tst-join7mod.so", RTLD_NOW | RTLD_GLOBAL);
  if (f)
    dlclose (f);
  else
    return 1;

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
