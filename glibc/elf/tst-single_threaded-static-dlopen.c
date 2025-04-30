/* Test support for single-thread optimizations.  No threads, static dlopen.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* In a static dlopen scenario, the single-threaded optimization is
   not possible because their is no globally shared dynamic linker
   across all namespaces.  */

#include <stddef.h>
#include <support/check.h>
#include <support/xdlfcn.h>
#include <sys/single_threaded.h>

static int
do_test (void)
{
  TEST_VERIFY (__libc_single_threaded);

  /* Defined in tst-single-threaded-mod1.o.  */
  extern _Bool single_threaded_1 (void);
  TEST_VERIFY (single_threaded_1 ());

  /* A failed dlopen does not change the multi-threaded status.  */
  TEST_VERIFY (dlopen ("tst-single_threaded-does-not-exist.so", RTLD_LAZY)
               == NULL);
  TEST_VERIFY (__libc_single_threaded);
  TEST_VERIFY (single_threaded_1 ());

  /* And neither does a successful dlopen for outer (static) libc.  */
  void *handle_mod2 = xdlopen ("tst-single_threaded-mod2.so", RTLD_LAZY);
  _Bool (*single_threaded_2) (void)
    = xdlsym (handle_mod2, "single_threaded_2");
  TEST_VERIFY (__libc_single_threaded);
  TEST_VERIFY (single_threaded_1 ());
  /* The inner libc always assumes multi-threaded use.  */
  TEST_VERIFY (!single_threaded_2 ());

  xdlclose (handle_mod2);

  return 0;
}

#include <support/test-driver.c>
