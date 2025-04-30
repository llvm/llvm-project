/* Test with many dynamic TLS variables.
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

/* This test intends to exercise dynamic TLS variable allocation.  It
   achieves this by combining dlopen (to avoid static TLS allocation
   after static TLS resizing), many DSOs with a large variable (to
   exceed the static TLS reserve), and an already-running thread (to
   force full dynamic TLS initialization).  */

#include "tst-tls-manydynamic.h"

#include <errno.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int do_test (void);
#include <support/xthread.h>
#include <support/test-driver.c>

void *handles[COUNT];
set_value_func set_value_funcs[COUNT];
get_value_func get_value_funcs[COUNT];

static void
init_functions (void)
{
  for (int i = 0; i < COUNT; ++i)
    {
      /* Open the module.  */
      {
        char soname[100];
        snprintf (soname, sizeof (soname), "tst-tls-manydynamic%02dmod.so", i);
        handles[i] = dlopen (soname, RTLD_LAZY);
        if (handles[i] == NULL)
          {
            printf ("error: dlopen failed: %s\n", dlerror ());
            exit (1);
          }
      }

      /* Obtain the setter function.  */
      {
        char fname[100];
        snprintf (fname, sizeof (fname), "set_value_%02d", i);
        void *func = dlsym (handles[i], fname);
        if (func == NULL)
          {
            printf ("error: dlsym: %s\n", dlerror ());
            exit (1);
          }
        set_value_funcs[i] = func;
      }

      /* Obtain the getter function.  */
      {
        char fname[100];
        snprintf (fname, sizeof (fname), "get_value_%02d", i);
        void *func = dlsym (handles[i], fname);
        if (func == NULL)
          {
            printf ("error: dlsym: %s\n", dlerror ());
            exit (1);
          }
        get_value_funcs[i] = func;
      }
    }
}

static pthread_barrier_t barrier;

/* Running thread which forces real TLS initialization.  */
static void *
blocked_thread_func (void *closure)
{
  xpthread_barrier_wait (&barrier);

  /* TLS test runs here in the main thread.  */

  xpthread_barrier_wait (&barrier);
  return NULL;
}

static int
do_test (void)
{
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

  init_functions ();

  struct value values[COUNT];
  /* Initialze the TLS variables.  */
  for (int i = 0; i < COUNT; ++i)
    {
      for (int j = 0; j < PER_VALUE_COUNT; ++j)
        values[i].num[j] = rand ();
      set_value_funcs[i] (&values[i]);
    }

  /* Read back their values to check that they do not overlap.  */
  for (int i = 0; i < COUNT; ++i)
    {
      struct value actual;
      get_value_funcs[i] (&actual);

      for (int j = 0; j < PER_VALUE_COUNT; ++j)
        if (actual.num[j] != values[i].num[j])
        {
          printf ("error: mismatch at variable %d/%d: %d != %d\n",
                  i, j, actual.num[j], values[i].num[j]);
          exit (1);
        }
    }

  xpthread_barrier_wait (&barrier);
  xpthread_join (blocked_thread);

  /* Close the modules.  */
  for (int i = 0; i < COUNT; ++i)
    dlclose (handles[i]);

  return 0;
}
