/* Verify that DSO is unloaded only if its TLS objects are destroyed.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

/* For the default case, i.e. NO_DELETE not defined, the test dynamically loads
   a DSO and spawns a thread that subsequently calls into the DSO to register a
   destructor for an object in the DSO and then calls dlclose on the handle for
   the DSO.  When the thread exits, the DSO should not be unloaded or else the
   destructor called during thread exit will crash.  Further in the main
   thread, the DSO is opened and closed again, at which point the DSO should be
   unloaded.

   When NO_DELETE is defined, the DSO is loaded twice, once with just RTLD_LAZY
   flag and the second time with the RTLD_NODELETE flag set.  The thread is
   spawned, destructor registered and then thread exits without closing the
   DSO.  In the main thread, the first handle is then closed, followed by the
   second handle.  In the end, the DSO should remain loaded due to the
   RTLD_NODELETE flag being set in the second dlopen call.  */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <link.h>
#include <stdbool.h>
#include <support/xdlfcn.h>

#ifndef NO_DELETE
# define LOADED_IS_GOOD false
#endif

#ifndef H2_RTLD_FLAGS
# define H2_RTLD_FLAGS (RTLD_LAZY)
#endif

#define DSO_NAME "$ORIGIN/tst-tls-atexit-lib.so"

/* Walk through the map in the _r_debug structure to see if our lib is still
   loaded.  */
static bool
is_loaded (void)
{
  struct link_map *lm = (struct link_map *) _r_debug.r_map;

  for (; lm; lm = lm->l_next)
    if (lm->l_type == lt_loaded && lm->l_name
	&& strcmp (basename (DSO_NAME), basename (lm->l_name)) == 0)
      {
	printf ("%s is still loaded\n", lm->l_name);
	return true;
      }
  return false;
}

/* Accept a valid handle returned by DLOPEN, load the reg_dtor symbol to
   register a destructor and then call dlclose on the handle.  The dlclose
   should not unload the DSO since the destructor has not been called yet.  */
static void *
reg_dtor_and_close (void *h)
{
  void (*reg_dtor) (void) = (void (*) (void)) xdlsym (h, "reg_dtor");

  reg_dtor ();

#ifndef NO_DELETE
  xdlclose (h);
#endif

  return NULL;
}

static int
spawn_thread (void *h)
{
  pthread_t t;
  int ret;
  void *thr_ret;

  if ((ret = pthread_create (&t, NULL, reg_dtor_and_close, h)) != 0)
    {
      printf ("pthread_create failed: %s\n", strerror (ret));
      return 1;
    }

  if ((ret = pthread_join (t, &thr_ret)) != 0)
    {
      printf ("pthread_join failed: %s\n", strerror (ret));
      return 1;
    }

  if (thr_ret != NULL)
    return 1;

  return 0;
}

static int
do_test (void)
{
  /* Load the DSO.  */
  void *h1 = xdlopen (DSO_NAME, RTLD_LAZY);

#ifndef NO_DELETE
  if (spawn_thread (h1) != 0)
    return 1;
#endif

  void *h2 = xdlopen (DSO_NAME, H2_RTLD_FLAGS);

#ifdef NO_DELETE
  if (spawn_thread (h1) != 0)
    return 1;

  xdlclose (h1);
#endif
  xdlclose (h2);

  /* Check link maps to ensure that the DSO has unloaded.  In the normal case,
     the DSO should be unloaded if there are no uses.  However, if one of the
     dlopen calls were with RTLD_NODELETE, the DSO should remain loaded.  */
  return is_loaded () == LOADED_IS_GOOD ? 0 : 1;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
