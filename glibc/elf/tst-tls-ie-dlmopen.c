/* Test dlopen of modules with initial-exec TLS after dlmopen.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* This test tries to check that surplus static TLS is not used up for
   dynamic TLS optimizations and 4*144 = 576 bytes of static TLS is
   still available for dlopening modules with initial-exec TLS after 3
   new dlmopen namespaces are created.  It depends on rtld.nns=4 and
   rtld.optional_static_tls=512 tunable settings.  */

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int do_test (void);
#include <support/xthread.h>
#include <support/xdlfcn.h>
#include <support/check.h>
#include <support/test-driver.c>

/* Have some big TLS in the main exe: should not use surplus TLS.  */
__thread char maintls[1000];

static pthread_barrier_t barrier;

/* Forces multi-threaded behaviour.  */
static void *
blocked_thread_func (void *closure)
{
  xpthread_barrier_wait (&barrier);
  /* TLS load and access tests run here in the main thread.  */
  xpthread_barrier_wait (&barrier);
  return NULL;
}

static void *
load_and_access (Lmid_t lmid, const char *mod, const char *func)
{
  /* Load module with TLS.  */
  void *p = xdlmopen (lmid, mod, RTLD_NOW);
  /* Access the TLS variable to ensure it is allocated.  */
  void (*f) (void) = (void (*) (void))xdlsym (p, func);
  f ();
  return p;
}

static int
do_test (void)
{
  void *mods[5];

  {
    int ret = pthread_barrier_init (&barrier, NULL, 2);
    if (ret != 0)
      {
        errno = ret;
        printf ("error: pthread_barrier_init: %m\n");
        exit (1);
      }
  }

  pthread_t blocked_thread = xpthread_create (NULL, blocked_thread_func, NULL);
  xpthread_barrier_wait (&barrier);

  printf ("maintls[%zu]:\t %p .. %p\n",
	   sizeof maintls, maintls, maintls + sizeof maintls);
  memset (maintls, 1, sizeof maintls);

  /* Load modules with dynamic TLS (use surplus static TLS for libc
     in new namespaces and may be for TLS optimizations too).  */
  mods[0] = load_and_access (LM_ID_BASE, "tst-tls-ie-mod0.so", "access0");
  mods[1] = load_and_access (LM_ID_NEWLM, "tst-tls-ie-mod1.so", "access1");
  mods[2] = load_and_access (LM_ID_NEWLM, "tst-tls-ie-mod2.so", "access2");
  mods[3] = load_and_access (LM_ID_NEWLM, "tst-tls-ie-mod3.so", "access3");
  /* Load modules with initial-exec TLS (can only use surplus static TLS).  */
  mods[4] = load_and_access (LM_ID_BASE, "tst-tls-ie-mod6.so", "access6");

  /* Here 576 bytes + 3 * libc use of surplus static TLS is in use so less
     than 1024 bytes are available (exact number depends on TLS optimizations
     and the libc TLS use).  */
  printf ("The next dlmopen should fail...\n");
  void *p = dlmopen (LM_ID_BASE, "tst-tls-ie-mod4.so", RTLD_NOW);
  if (p != NULL)
    FAIL_EXIT1 ("error: expected dlmopen to fail because there is "
		"not enough surplus static TLS.\n");
  printf ("...OK failed with: %s.\n", dlerror ());

  xpthread_barrier_wait (&barrier);
  xpthread_join (blocked_thread);

  /* Close the modules.  */
  for (int i = 0; i < 5; ++i)
    xdlclose (mods[i]);

  return 0;
}
