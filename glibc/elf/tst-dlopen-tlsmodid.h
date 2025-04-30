/* Common code for tst-dlopen-tlsmodid, tst-dlopen-tlsmodid-pie,
   tst-dlopen-tlsmodid-container.

   Verify that incorrectly dlopen()ing an executable without
   __RTLD_OPENEXEC does not cause assertion in ld.so, and that it
   actually results in an error.

   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

/* Before including this file, the macro TST_DLOPEN_TLSMODID_PATH must
   be defined, to specify the path used for the open operation.  */

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xthread.h>

__thread int x;

void *
fn (void *p)
{
  return p;
}

/* Call dlopen and check that fails with an error message indicating
   an attempt to open an ET_EXEC or PIE object.  */
static void
check_dlopen_failure (void)
{
  void *handle = dlopen (TST_DLOPEN_TLSMODID_PATH, RTLD_LAZY);
  if (handle != NULL)
    FAIL_EXIT1 ("dlopen succeeded unexpectedly: %s", TST_DLOPEN_TLSMODID_PATH);

  const char *message = dlerror ();
  TEST_VERIFY_EXIT (message != NULL);
  if ((strstr (message,
	       "cannot dynamically load position-independent executable")
       == NULL)
      && strstr (message, "cannot dynamically load executable") == NULL)
    FAIL_EXIT1 ("invalid dlopen error message: \"%s\"", message);
}

static int
do_test (int argc, char *argv[])
{
  int j;

  for (j = 0; j < 100; ++j)
    {
      pthread_t thr;

      check_dlopen_failure ();

      /* We create threads to force TLS allocation, which triggers
	 the original bug i.e. running out of surplus slotinfo entries
	 for TLS.  */
      thr = xpthread_create (NULL, fn, NULL);
      xpthread_join (thr);
    }

  check_dlopen_failure ();

  return 0;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
