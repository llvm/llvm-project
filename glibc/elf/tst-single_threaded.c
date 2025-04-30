/* Test support for single-thread optimizations.  No threads.
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

#include <stddef.h>
#include <stdio.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xdlfcn.h>
#include <sys/single_threaded.h>

/* Defined in tst-single-threaded-mod1.so.  */
extern _Bool single_threaded_1 (void);

/* Initialized via dlsym.  */
_Bool (*single_threaded_2) (void);
_Bool (*single_threaded_3) (void);

static void
subprocess (void *closure)
{
  TEST_VERIFY (__libc_single_threaded);
  TEST_VERIFY (single_threaded_1 ());
  if (single_threaded_2 != NULL)
    TEST_VERIFY (single_threaded_2 ());
  if (single_threaded_3 != NULL)
    TEST_VERIFY (!single_threaded_3 ());
}

static int
do_test (void)
{
  TEST_VERIFY (__libc_single_threaded);
  TEST_VERIFY (single_threaded_1 ());
  support_isolate_in_subprocess (subprocess, NULL);

  void *handle_mod2 = xdlopen ("tst-single_threaded-mod2.so", RTLD_LAZY);
  single_threaded_2 = xdlsym (handle_mod2, "single_threaded_2");
  TEST_VERIFY (single_threaded_2 ());
  support_isolate_in_subprocess (subprocess, NULL);

  /* The current implementation treats the inner namespace as
     multi-threaded.  */
  void *handle_mod3 = dlmopen (LM_ID_NEWLM, "tst-single_threaded-mod3.so",
                               RTLD_LAZY);
  single_threaded_3 = xdlsym (handle_mod3, "single_threaded_3");
  TEST_VERIFY (!single_threaded_3 ());
  support_isolate_in_subprocess (subprocess, NULL);

  xdlclose (handle_mod3);
  xdlclose (handle_mod2);

  return 0;
}

#include <support/test-driver.c>
